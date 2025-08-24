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
        item_history = inputs["item_history_embeddings"]  # [batch_size, seq_len, emb_dim]
        
        # Process demographics through embeddings
        age_emb = self.age_embedding(age)  # [batch_size, embedding_dim//16]
        income_emb = self.income_embedding(income)  # [batch_size, embedding_dim//16]
        gender_emb = self.gender_embedding(gender)  # [batch_size, embedding_dim//16]
        
        # Aggregate item history using attention
        # Create attention mask for padding
        history_mask = tf.reduce_sum(tf.abs(item_history), axis=-1) > 0  # [batch_size, seq_len]
        
        # Self-attention on history (remove attention_mask due to shape issues)
        attended_history = self.history_attention(
            query=item_history,
            value=item_history,
            key=item_history,
            training=training
        )
        
        # Mean pooling over history length
        history_aggregated = tf.reduce_mean(attended_history, axis=1)
        
        # Combine all features
        combined = tf.concat([
            age_emb,
            income_emb,
            gender_emb,
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