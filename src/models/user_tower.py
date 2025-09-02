import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


class UserTower(tf.keras.Model):
    """User tower for two-tower recommendation architecture."""
    
    def __init__(self,
                 max_history_length: int = 50,
                 embedding_dim: int = 128,  # Output embedding dimension
                 hidden_dims: list = [256, 128],  # Internal dims for processing
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length
        
        # Demographic embeddings (categorical features)
        # Age: 6 categories (Teen, Young Adult, Adult, Middle Age, Mature, Senior)
        self.age_embedding = tf.keras.layers.Embedding(
            6, embedding_dim // 16, name="age_embedding"
        )
        
        # Income: 5 categories (percentile-based)
        self.income_embedding = tf.keras.layers.Embedding(
            5, embedding_dim // 16, name="income_embedding"
        )
        
        # Gender: 2 categories (0=female, 1=male)
        self.gender_embedding = tf.keras.layers.Embedding(
            2, embedding_dim // 16, name="gender_embedding"
        )
        
        # New demographic embeddings
        # Profession: 8 categories (Technology, Healthcare, Education, Finance, Retail, Manufacturing, Services, Other)
        self.profession_embedding = tf.keras.layers.Embedding(
            8, embedding_dim // 16, name="profession_embedding"
        )
        
        # Location: 3 categories (Urban, Suburban, Rural)
        self.location_embedding = tf.keras.layers.Embedding(
            3, embedding_dim // 16, name="location_embedding"
        )
        
        # Education Level: 5 categories (High School, Some College, Bachelor's, Master's, PhD+)
        self.education_embedding = tf.keras.layers.Embedding(
            5, embedding_dim // 16, name="education_embedding"
        )
        
        # Marital Status: 4 categories (Single, Married, Divorced, Widowed)
        self.marital_embedding = tf.keras.layers.Embedding(
            4, embedding_dim // 16, name="marital_embedding"
        )
        
        # History aggregation layers
        self.history_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=embedding_dim,
            name="history_attention"
        )
        
        # Combine demographics and history
        self.dense_layers = []
        for i, dim in enumerate(hidden_dims):
            self.dense_layers.extend([
                tf.keras.layers.Dense(dim, activation="relu", name=f"user_dense_{i}"),
                tf.keras.layers.Dropout(dropout_rate, name=f"user_dropout_{i}")
            ])
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            embedding_dim, activation=None, name="user_output"
        )
        
    def call(self, inputs, training=None):
        """Forward pass of the user tower."""
        age = inputs["age"]  # Now categorical (0-5)
        gender = inputs["gender"]  # Categorical (0-1)
        income = inputs["income"]  # Now categorical (0-4)
        profession = inputs["profession"]  # Categorical (0-7)
        location = inputs["location"]  # Categorical (0-2)
        education = inputs["education_level"]  # Categorical (0-4)
        marital_status = inputs["marital_status"]  # Categorical (0-3)
        item_history = inputs["item_history_embeddings"]  # [batch_size, seq_len, emb_dim]
        
        # Process demographics through embeddings
        age_emb = self.age_embedding(age)  # [batch_size, embedding_dim//16]
        income_emb = self.income_embedding(income)  # [batch_size, embedding_dim//16]
        gender_emb = self.gender_embedding(gender)  # [batch_size, embedding_dim//16]
        profession_emb = self.profession_embedding(profession)  # [batch_size, embedding_dim//16]
        location_emb = self.location_embedding(location)  # [batch_size, embedding_dim//16]
        education_emb = self.education_embedding(education)  # [batch_size, embedding_dim//16]
        marital_emb = self.marital_embedding(marital_status)  # [batch_size, embedding_dim//16]
        
        # Aggregate item history using attention
        # Create attention mask for padding
        history_mask = tf.reduce_sum(tf.abs(item_history), axis=-1) > 0  # [batch_size, seq_len]
        
        # Check if users have any interactions at all
        has_any_interactions = tf.reduce_any(history_mask, axis=1)  # [batch_size]
        
        # For users with interactions: apply attention mechanism
        # Reshape mask for MultiHeadAttention: [batch_size, 1, seq_len] -> broadcasts to [batch_size, seq_len, seq_len]
        attention_mask = tf.expand_dims(history_mask, axis=1)  # [batch_size, 1, seq_len]
        
        # Self-attention on history with proper masking
        attended_history = self.history_attention(
            query=item_history,
            value=item_history,
            key=item_history,
            attention_mask=attention_mask,
            training=training
        )
        
        # Masked mean pooling over history length (only average over non-padding tokens)
        history_aggregated = self._masked_mean_pooling(attended_history, history_mask)
        
        # For zero-interaction users, history_aggregated will be all zeros due to masked pooling
        # This is correct behavior - they should rely entirely on demographic features
        
        # Combine all features
        combined = tf.concat([
            age_emb,
            income_emb,
            gender_emb,
            profession_emb,
            location_emb,
            education_emb,
            marital_emb,
            history_aggregated
        ], axis=-1)
        
        # Pass through dense layers
        x = combined
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        # Final output
        output = self.output_layer(x)
        
        # L2 normalize for similarity computations
        return tf.nn.l2_normalize(output, axis=-1)
    
    def _masked_mean_pooling(self, sequence: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Perform masked mean pooling over sequence dimension.
        
        Args:
            sequence: [batch_size, seq_len, embedding_dim]
            mask: [batch_size, seq_len] - True for valid positions, False for padding
            
        Returns:
            pooled: [batch_size, embedding_dim]
        """
        # Convert mask to float and add dimension for broadcasting
        mask_float = tf.cast(mask, tf.float32)  # [batch_size, seq_len]
        mask_expanded = tf.expand_dims(mask_float, axis=-1)  # [batch_size, seq_len, 1]
        
        # Apply mask to sequence (zero out padding positions)
        masked_sequence = sequence * mask_expanded  # [batch_size, seq_len, embedding_dim]
        
        # Sum over sequence dimension
        sequence_sum = tf.reduce_sum(masked_sequence, axis=1)  # [batch_size, embedding_dim]
        
        # Count valid (non-padding) positions per batch item
        valid_counts = tf.reduce_sum(mask_float, axis=1, keepdims=True)  # [batch_size, 1]
        
        # Avoid division by zero for users with no interactions
        valid_counts = tf.maximum(valid_counts, 1.0)
        
        # Compute mean only over valid positions
        pooled = sequence_sum / valid_counts  # [batch_size, embedding_dim]
        
        return pooled


class TwoTowerModel(tfrs.Model):
    """Complete two-tower recommendation model."""
    
    def __init__(self, 
                 item_tower: tf.keras.Model,
                 user_tower: UserTower,
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0):
        super().__init__()
        
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        
        # Rating prediction task
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Rating task
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        # Retrieval loss
        self.retrieval_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
    def call(self, features):
        user_embeddings = self.user_tower(features)
        positive_item_embeddings = self.item_tower(features)
        
        return {
            "user_embedding": user_embeddings,
            "item_embedding": positive_item_embeddings
        }
    
    def compute_loss(self, features, training=False):
        user_embeddings = self.user_tower(features)
        positive_item_embeddings = self.item_tower(features)
        
        # Rating prediction
        concatenated = tf.concat([user_embeddings, positive_item_embeddings], axis=-1)
        rating_predictions = self.rating_model(concatenated)
        
        # Rating loss
        rating_loss = self.rating_task(
            labels=features["rating"],
            predictions=rating_predictions
        )
        
        # Retrieval loss - dot product similarity
        similarities = tf.reduce_sum(user_embeddings * positive_item_embeddings, axis=1)
        retrieval_loss = self.retrieval_loss(features["rating"], tf.nn.sigmoid(similarities))
        
        # Combine losses
        total_loss = (
            self.rating_weight * rating_loss +
            self.retrieval_weight * retrieval_loss
        )
        
        # Return scalar loss for TFX compatibility
        return total_loss