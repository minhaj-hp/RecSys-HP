#!/usr/bin/env python3
"""
Improved two-tower model with better embedding discrimination and training stability.
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


class ImprovedItemTower(tf.keras.Model):
    """Enhanced item tower with better discrimination and representation capacity."""
    
    def __init__(self, 
                 item_vocab_size: int,
                 category_vocab_size: int,
                 brand_vocab_size: int,
                 embedding_dim: int = 128,  # Increased from 64
                 hidden_dims: list = [256, 128],  # Deeper network
                 dropout_rate: float = 0.3,
                 use_bias: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        
        # Larger embedding layers with proper initialization
        self.item_embedding = tf.keras.layers.Embedding(
            item_vocab_size, embedding_dim, 
            embeddings_initializer='glorot_uniform',
            name="item_embedding"
        )
        self.category_embedding = tf.keras.layers.Embedding(
            category_vocab_size, embedding_dim,
            embeddings_initializer='glorot_uniform', 
            name="category_embedding"
        )
        self.brand_embedding = tf.keras.layers.Embedding(
            brand_vocab_size, embedding_dim,
            embeddings_initializer='glorot_uniform',
            name="brand_embedding"
        )
        
        # Price normalization and projection
        self.price_normalization = tf.keras.layers.Normalization(name="price_norm")
        self.price_projection = tf.keras.layers.Dense(
            embedding_dim // 4, activation='relu', name="price_proj"
        )
        
        # Attention mechanism for feature fusion
        self.feature_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=embedding_dim,
            name="feature_attention"
        )
        
        # Enhanced dense layers with batch normalization
        self.dense_layers = []
        for i, dim in enumerate(hidden_dims):
            self.dense_layers.extend([
                tf.keras.layers.Dense(dim, activation=None, name=f"dense_{i}"),
                tf.keras.layers.BatchNormalization(name=f"bn_{i}"),
                tf.keras.layers.Activation('relu', name=f"relu_{i}"),
                tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
            ])
        
        # Output projection with bias term
        self.output_layer = tf.keras.layers.Dense(
            embedding_dim, activation=None, use_bias=use_bias, name="item_output"
        )
        
        # Learnable bias term for each item
        if use_bias:
            self.item_bias = tf.keras.layers.Embedding(
                item_vocab_size, 1, name="item_bias"
            )
        
    def call(self, inputs, training=None):
        """Enhanced forward pass with attention and better feature fusion."""
        item_id = inputs["product_id"]
        category_id = inputs["category_id"] 
        brand_id = inputs["brand_id"]
        price = inputs["price"]
        
        # Get embeddings
        item_emb = self.item_embedding(item_id)  # [batch, emb_dim]
        category_emb = self.category_embedding(category_id)
        brand_emb = self.brand_embedding(brand_id)
        
        # Process price
        price_norm = self.price_normalization(tf.expand_dims(price, -1))
        price_emb = self.price_projection(price_norm)
        
        # Pad price embedding to match others
        price_emb_padded = tf.pad(
            price_emb, 
            [[0, 0], [0, self.embedding_dim - tf.shape(price_emb)[-1]]]
        )
        
        # Stack features for attention [batch, 4, emb_dim]
        features = tf.stack([item_emb, category_emb, brand_emb, price_emb_padded], axis=1)
        
        # Apply self-attention for feature fusion
        attended_features = self.feature_attention(
            query=features,
            value=features,
            key=features,
            training=training
        )
        
        # Aggregate features (mean pooling)
        combined = tf.reduce_mean(attended_features, axis=1)
        
        # Pass through enhanced dense layers
        x = combined
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        # Final output
        output = self.output_layer(x)
        
        # L2 normalize for similarity computations
        normalized_output = tf.nn.l2_normalize(output, axis=-1)
        
        # Add bias if enabled
        if self.use_bias:
            bias = tf.squeeze(self.item_bias(item_id), axis=-1)
            return normalized_output, bias
        else:
            return normalized_output


class ImprovedUserTower(tf.keras.Model):
    """Enhanced user tower with better history modeling and representation."""
    
    def __init__(self,
                 max_history_length: int = 50,
                 embedding_dim: int = 128,  # Increased from 64
                 hidden_dims: list = [256, 128],  # Deeper network
                 dropout_rate: float = 0.3,
                 use_bias: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length
        self.use_bias = use_bias
        
        # Demographic embeddings (categorical features)
        # Age: 6 categories (Teen, Young Adult, Adult, Middle Age, Mature, Senior)
        self.age_embedding = tf.keras.layers.Embedding(
            6, embedding_dim // 16,
            embeddings_initializer='glorot_uniform',
            name="age_embedding"
        )
        
        # Income: 5 categories (percentile-based)
        self.income_embedding = tf.keras.layers.Embedding(
            5, embedding_dim // 16,
            embeddings_initializer='glorot_uniform',
            name="income_embedding"
        )
        
        # Gender: 2 categories (0=female, 1=male)
        self.gender_embedding = tf.keras.layers.Embedding(
            2, embedding_dim // 16, 
            embeddings_initializer='glorot_uniform',
            name="gender_embedding"
        )
        
        # Improved history processing with positional encoding
        self.history_transformer = tf.keras.layers.MultiHeadAttention(
            num_heads=8,  # More attention heads
            key_dim=embedding_dim,
            name="history_transformer"
        )
        
        # History aggregation with learned weights
        self.history_aggregation = tf.keras.layers.Dense(
            embedding_dim, activation='tanh', name="history_agg"
        )
        
        # Enhanced dense layers with batch normalization
        self.dense_layers = []
        for i, dim in enumerate(hidden_dims):
            self.dense_layers.extend([
                tf.keras.layers.Dense(dim, activation=None, name=f"user_dense_{i}"),
                tf.keras.layers.BatchNormalization(name=f"user_bn_{i}"),
                tf.keras.layers.Activation('relu', name=f"user_relu_{i}"),
                tf.keras.layers.Dropout(dropout_rate, name=f"user_dropout_{i}")
            ])
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            embedding_dim, activation=None, use_bias=use_bias, name="user_output"
        )
        
        # Learnable user bias
        if use_bias:
            # We'll need to handle user bias differently since we don't have user vocab in inference
            self.global_user_bias = tf.Variable(
                initial_value=0.0, trainable=True, name="global_user_bias"
            )
        
    def call(self, inputs, training=None):
        """Enhanced forward pass with better history modeling."""
        age = inputs["age"]  # Now categorical (0-5)
        gender = inputs["gender"]  # Categorical (0-1)
        income = inputs["income"]  # Now categorical (0-4)
        item_history = inputs["item_history_embeddings"]  # [batch_size, seq_len, emb_dim]
        
        # Process demographics through embeddings
        age_emb = self.age_embedding(age)  # [batch_size, embedding_dim//16]
        income_emb = self.income_embedding(income)  # [batch_size, embedding_dim//16]
        gender_emb = self.gender_embedding(gender)  # [batch_size, embedding_dim//16]
        
        # Combine all demographic embeddings
        demo_combined = tf.concat([age_emb, income_emb, gender_emb], axis=-1)
        # Total demographics: 3 * (embedding_dim//16) = ~18.75% of embedding_dim
        
        # Enhanced history processing with positional encoding
        batch_size = tf.shape(item_history)[0]
        seq_len = tf.shape(item_history)[1]
        
        # Create positional encoding
        positions = tf.range(seq_len, dtype=tf.float32)
        position_encoding = tf.sin(
            positions[:, tf.newaxis] / 
            tf.pow(10000.0, 2 * tf.range(self.embedding_dim, dtype=tf.float32) / self.embedding_dim)
        )
        position_encoding = tf.expand_dims(position_encoding, 0)
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
        
        # Add positional encoding to history
        history_with_pos = item_history + position_encoding
        
        # Create attention mask for padding
        history_mask = tf.reduce_sum(tf.abs(item_history), axis=-1) > 0
        
        # Apply transformer attention to history
        attended_history = self.history_transformer(
            query=history_with_pos,
            value=history_with_pos,
            key=history_with_pos,
            attention_mask=history_mask,
            training=training
        )
        
        # Aggregate history with learned weights
        history_weights = tf.nn.softmax(
            tf.keras.layers.Dense(1)(attended_history), axis=1
        )
        history_aggregated = tf.reduce_sum(
            attended_history * history_weights, axis=1
        )
        
        # Apply additional processing
        history_processed = self.history_aggregation(history_aggregated)
        
        # Combine all features
        combined = tf.concat([
            demo_combined,
            history_processed
        ], axis=-1)
        
        # Pass through enhanced dense layers
        x = combined
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        # Final output
        output = self.output_layer(x)
        
        # L2 normalize for similarity computations
        normalized_output = tf.nn.l2_normalize(output, axis=-1)
        
        # Add global bias if enabled
        if self.use_bias:
            return normalized_output, self.global_user_bias
        else:
            return normalized_output


class TemperatureScaledSimilarity(tf.keras.layers.Layer):
    """Learnable temperature scaling for similarity computations."""
    
    def __init__(self, initial_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_temperature = initial_temperature
        
    def build(self, input_shape):
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_temperature),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, user_embeddings, item_embeddings):
        """Compute temperature-scaled similarity."""
        # Dot product similarity
        similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
        
        # Scale by learnable temperature
        scaled_similarities = similarities / tf.maximum(self.temperature, 0.01)  # Prevent division by 0
        
        return scaled_similarities


class ImprovedTwoTowerModel(tfrs.Model):
    """Enhanced two-tower model with better discrimination and training stability."""
    
    def __init__(self, 
                 item_tower: ImprovedItemTower,
                 user_tower: ImprovedUserTower,
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0,
                 contrastive_weight: float = 0.5,
                 use_focal_loss: bool = True):
        super().__init__()
        
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        self.contrastive_weight = contrastive_weight
        self.use_focal_loss = use_focal_loss
        
        # Temperature-scaled similarity
        self.temperature_similarity = TemperatureScaledSimilarity()
        
        # Enhanced rating prediction with more capacity
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Rating task with better loss
        if use_focal_loss:
            self.rating_loss = self._focal_loss
        else:
            self.rating_loss = tf.keras.losses.BinaryCrossentropy()
        
        # Contrastive loss for embedding separation
        self.contrastive_loss = tf.keras.losses.CosineSimilarity()
        
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for handling imbalanced data."""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute focal weight
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Compute loss
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        focal_loss = focal_weight * bce
        
        return tf.reduce_mean(focal_loss)
    
    def call(self, features):
        # Get embeddings (handle bias if present)
        user_output = self.user_tower(features)
        item_output = self.item_tower(features)
        
        # Handle bias terms
        if isinstance(user_output, tuple):
            user_embeddings, user_bias = user_output
        else:
            user_embeddings = user_output
            user_bias = 0.0
            
        if isinstance(item_output, tuple):
            item_embeddings, item_bias = item_output
        else:
            item_embeddings = item_output
            item_bias = 0.0
        
        return {
            "user_embedding": user_embeddings,
            "item_embedding": item_embeddings,
            "user_bias": user_bias,
            "item_bias": item_bias
        }
    
    def _hard_negative_mining(self, user_embeddings, item_embeddings, ratings, num_negatives=5):
        """Mine hard negatives for better training."""
        batch_size = tf.shape(user_embeddings)[0]
        
        # Compute all pairwise similarities
        user_norm = tf.nn.l2_normalize(user_embeddings, axis=1)
        item_norm = tf.nn.l2_normalize(item_embeddings, axis=1)
        
        # Expand dimensions for broadcasting: [batch, 1, dim] x [1, batch, dim]
        user_expanded = tf.expand_dims(user_norm, 1)
        item_expanded = tf.expand_dims(item_norm, 0)
        
        # Compute similarity matrix [batch, batch]
        similarity_matrix = tf.reduce_sum(user_expanded * item_expanded, axis=2)
        
        # Create mask to exclude positive pairs
        positive_mask = tf.eye(batch_size, dtype=tf.bool)
        negative_mask = tf.logical_not(positive_mask)
        
        # Get negative similarities and find hardest negatives
        negative_similarities = tf.where(negative_mask, similarity_matrix, -tf.float32.max)
        
        # Get top-k hardest negatives (highest similarities among negatives)
        _, hard_negative_indices = tf.nn.top_k(negative_similarities, k=num_negatives)
        
        return hard_negative_indices

    def compute_loss(self, features, training=False):
        # Get embeddings and biases
        outputs = self(features)
        user_embeddings = outputs["user_embedding"]
        item_embeddings = outputs["item_embedding"]
        user_bias = outputs["user_bias"]
        item_bias = outputs["item_bias"]
        
        # Rating prediction with bias terms
        concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
        rating_predictions = self.rating_model(concatenated, training=training)
        
        # Add bias terms to rating predictions
        rating_predictions_with_bias = rating_predictions + user_bias + item_bias
        rating_predictions_with_bias = tf.nn.sigmoid(rating_predictions_with_bias)
        
        # Rating loss
        rating_loss = self.rating_loss(features["rating"], rating_predictions_with_bias)
        
        # Temperature-scaled retrieval loss
        scaled_similarities = self.temperature_similarity(user_embeddings, item_embeddings)
        retrieval_loss = tf.keras.losses.binary_crossentropy(
            features["rating"], 
            tf.nn.sigmoid(scaled_similarities)
        )
        retrieval_loss = tf.reduce_mean(retrieval_loss)
        
        # Enhanced contrastive loss with hard negative mining
        batch_size = tf.shape(user_embeddings)[0]
        
        if training and batch_size > 5:  # Only use hard negatives during training with sufficient batch size
            # Hard negative mining
            hard_negative_indices = self._hard_negative_mining(
                user_embeddings, item_embeddings, features["rating"], num_negatives=3
            )
            
            # Positive similarities
            positive_similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
            
            # Hard negative similarities
            hard_negative_losses = []
            for i in range(3):  # Use top 3 hard negatives
                neg_indices = hard_negative_indices[:, i]
                negative_item_embeddings = tf.gather(item_embeddings, neg_indices)
                negative_similarities = tf.reduce_sum(user_embeddings * negative_item_embeddings, axis=1)
                
                # Triplet-like loss with margin
                margin_loss = tf.maximum(0.0, 0.2 + negative_similarities - positive_similarities)
                hard_negative_losses.append(margin_loss)
            
            # Average hard negative losses
            contrastive_loss = tf.reduce_mean(tf.stack(hard_negative_losses))
            
        else:
            # Fallback to random negative sampling
            shuffled_indices = tf.random.shuffle(tf.range(batch_size))
            negative_item_embeddings = tf.gather(item_embeddings, shuffled_indices)
            
            # Positive similarities
            positive_similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
            
            # Negative similarities  
            negative_similarities = tf.reduce_sum(user_embeddings * negative_item_embeddings, axis=1)
            
            # Contrastive loss (maximize positive, minimize negative)
            contrastive_loss = tf.reduce_mean(
                tf.maximum(0.0, 0.5 + negative_similarities - positive_similarities)
            )
        
        # Combine losses
        total_loss = (
            self.rating_weight * rating_loss +
            self.retrieval_weight * retrieval_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        # Add L2 regularization to prevent overfitting
        l2_loss = tf.add_n([
            tf.nn.l2_loss(var) for var in self.trainable_variables
            if 'bias' not in var.name and 'normalization' not in var.name
        ]) * 1e-5
        
        total_loss += l2_loss
        
        return {
            'total_loss': total_loss,
            'rating_loss': rating_loss,
            'retrieval_loss': retrieval_loss,
            'contrastive_loss': contrastive_loss,
            'l2_loss': l2_loss
        }


def create_improved_model(data_processor, 
                         embedding_dim=128,
                         use_bias=True,
                         use_focal_loss=True):
    """Factory function to create improved two-tower model."""
    
    # Create enhanced towers
    item_tower = ImprovedItemTower(
        item_vocab_size=len(data_processor.item_vocab),
        category_vocab_size=len(data_processor.category_vocab),
        brand_vocab_size=len(data_processor.brand_vocab),
        embedding_dim=embedding_dim,
        use_bias=use_bias
    )
    
    user_tower = ImprovedUserTower(
        max_history_length=50,
        embedding_dim=embedding_dim,
        use_bias=use_bias
    )
    
    # Create improved model
    model = ImprovedTwoTowerModel(
        item_tower=item_tower,
        user_tower=user_tower,
        rating_weight=1.0,
        retrieval_weight=0.5,
        contrastive_weight=0.3,
        use_focal_loss=use_focal_loss
    )
    
    return model