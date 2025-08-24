#!/usr/bin/env python3
"""
Enhanced two-tower model with embedding diversity regularization and improved discrimination.
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


class EmbeddingDiversityRegularizer(tf.keras.layers.Layer):
    """Regularizer to prevent embedding collapse by enforcing diversity."""
    
    def __init__(self, diversity_weight=0.01, orthogonality_weight=0.05, **kwargs):
        super().__init__(**kwargs)
        self.diversity_weight = diversity_weight
        self.orthogonality_weight = orthogonality_weight
    
    def call(self, embeddings):
        """Apply diversity regularization to embeddings."""
        batch_size = tf.shape(embeddings)[0]
        
        # Compute pairwise cosine similarities
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        similarity_matrix = tf.linalg.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True
        )
        
        # Remove diagonal (self-similarities)
        mask = 1.0 - tf.eye(batch_size)
        masked_similarities = similarity_matrix * mask
        
        # Diversity loss: penalize high similarities between different embeddings
        diversity_loss = tf.reduce_mean(tf.square(masked_similarities))
        
        # Orthogonality loss: encourage embeddings to be orthogonal
        identity_target = tf.eye(batch_size)
        orthogonality_loss = tf.reduce_mean(
            tf.square(similarity_matrix - identity_target)
        )
        
        # Add as regularization losses
        self.add_loss(self.diversity_weight * diversity_loss)
        self.add_loss(self.orthogonality_weight * orthogonality_loss)
        
        return embeddings


class AdaptiveTemperatureScaling(tf.keras.layers.Layer):
    """Advanced temperature scaling with learned parameters."""
    
    def __init__(self, initial_temperature=1.0, min_temp=0.1, max_temp=5.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_temperature = initial_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp
        
    def build(self, input_shape):
        # Learnable temperature with constraints
        self.raw_temperature = self.add_weight(
            name='raw_temperature',
            shape=(),
            initializer=tf.keras.initializers.Constant(
                np.log(self.initial_temperature - self.min_temp)
            ),
            trainable=True
        )
        
        # Learnable bias term for better discrimination
        self.similarity_bias = self.add_weight(
            name='similarity_bias',
            shape=(),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, user_embeddings, item_embeddings):
        """Compute adaptive temperature-scaled similarity with bias."""
        # Constrain temperature to valid range
        temperature = self.min_temp + tf.nn.softplus(self.raw_temperature)
        temperature = tf.minimum(temperature, self.max_temp)
        
        # Compute similarities
        similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
        
        # Add learnable bias and apply temperature scaling
        scaled_similarities = (similarities + self.similarity_bias) / temperature
        
        return scaled_similarities, temperature


class EnhancedItemTower(tf.keras.Model):
    """Enhanced item tower with diversity regularization."""
    
    def __init__(self, 
                 item_vocab_size: int,
                 category_vocab_size: int,
                 brand_vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout_rate: float = 0.3,
                 use_bias: bool = True,
                 use_diversity_reg: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias
        self.use_diversity_reg = use_diversity_reg
        
        # Embedding layers with better initialization
        self.item_embedding = tf.keras.layers.Embedding(
            item_vocab_size, embedding_dim,
            embeddings_initializer='he_normal',  # Better initialization
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="item_embedding"
        )
        self.category_embedding = tf.keras.layers.Embedding(
            category_vocab_size, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="category_embedding"
        )
        self.brand_embedding = tf.keras.layers.Embedding(
            brand_vocab_size, embedding_dim,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="brand_embedding"
        )
        
        # Price processing
        self.price_normalization = tf.keras.layers.Normalization(name="price_norm")
        self.price_projection = tf.keras.layers.Dense(
            embedding_dim // 4, activation='relu', name="price_proj"
        )
        
        # Enhanced attention mechanism
        self.feature_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=embedding_dim,
            dropout=0.1,
            name="feature_attention"
        )
        
        # Dense layers with residual connections
        self.dense_layers = []
        for i, dim in enumerate(hidden_dims):
            self.dense_layers.extend([
                tf.keras.layers.Dense(dim, activation=None, name=f"dense_{i}"),
                tf.keras.layers.BatchNormalization(name=f"bn_{i}"),
                tf.keras.layers.Activation('relu', name=f"relu_{i}"),
                tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
            ])
        
        # Output layer with controlled normalization
        self.output_layer = tf.keras.layers.Dense(
            embedding_dim, activation=None, use_bias=use_bias, name="item_output"
        )
        
        # Diversity regularizer
        if use_diversity_reg:
            self.diversity_regularizer = EmbeddingDiversityRegularizer()
        
        # Adaptive normalization instead of hard L2 normalization
        self.adaptive_norm = tf.keras.layers.LayerNormalization(name="adaptive_norm")
        
        # Item bias
        if use_bias:
            self.item_bias = tf.keras.layers.Embedding(
                item_vocab_size, 1, name="item_bias"
            )
    
    def call(self, inputs, training=None):
        """Enhanced forward pass with diversity regularization."""
        item_id = inputs["product_id"]
        category_id = inputs["category_id"] 
        brand_id = inputs["brand_id"]
        price = inputs["price"]
        
        # Get embeddings
        item_emb = self.item_embedding(item_id)
        category_emb = self.category_embedding(category_id)
        brand_emb = self.brand_embedding(brand_id)
        
        # Process price
        price_norm = self.price_normalization(tf.expand_dims(price, -1))
        price_emb = self.price_projection(price_norm)
        
        # Pad price embedding
        price_emb_padded = tf.pad(
            price_emb, 
            [[0, 0], [0, self.embedding_dim - tf.shape(price_emb)[-1]]]
        )
        
        # Stack features for attention
        features = tf.stack([item_emb, category_emb, brand_emb, price_emb_padded], axis=1)
        
        # Apply attention
        attended_features = self.feature_attention(
            query=features,
            value=features,
            key=features,
            training=training
        )
        
        # Aggregate with residual connection
        combined = tf.reduce_mean(attended_features + features, axis=1)
        
        # Pass through dense layers with residual connections
        x = combined
        residual = x
        for i, layer in enumerate(self.dense_layers):
            x = layer(x, training=training)
            # Add residual connection every 4 layers (complete block)
            if (i + 1) % 4 == 0 and x.shape[-1] == residual.shape[-1]:
                x = x + residual
                residual = x
        
        # Final output
        output = self.output_layer(x)
        
        # Apply diversity regularization if enabled
        if self.use_diversity_reg and training:
            output = self.diversity_regularizer(output)
        
        # Adaptive normalization instead of hard L2
        normalized_output = self.adaptive_norm(output)
        
        # Add bias if enabled
        if self.use_bias:
            bias = tf.squeeze(self.item_bias(item_id), axis=-1)
            return normalized_output, bias
        else:
            return normalized_output


class EnhancedUserTower(tf.keras.Model):
    """Enhanced user tower with diversity regularization."""
    
    def __init__(self,
                 max_history_length: int = 50,
                 embedding_dim: int = 128,
                 hidden_dims: list = [256, 128],
                 dropout_rate: float = 0.3,
                 use_bias: bool = True,
                 use_diversity_reg: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length
        self.use_bias = use_bias
        self.use_diversity_reg = use_diversity_reg
        
        # Demographic embeddings with regularization
        self.age_embedding = tf.keras.layers.Embedding(
            6, embedding_dim // 16,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="age_embedding"
        )
        self.income_embedding = tf.keras.layers.Embedding(
            5, embedding_dim // 16,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="income_embedding"
        )
        self.gender_embedding = tf.keras.layers.Embedding(
            2, embedding_dim // 16,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.L2(1e-6),
            name="gender_embedding"
        )
        
        # Enhanced history processing
        self.history_transformer = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=embedding_dim,
            dropout=0.1,
            name="history_transformer"
        )
        
        # History aggregation with attention pooling
        self.history_attention_pooling = tf.keras.layers.Dense(
            1, activation=None, name="history_attention"
        )
        
        # Dense layers with residual connections
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
        
        # Diversity regularizer
        if use_diversity_reg:
            self.diversity_regularizer = EmbeddingDiversityRegularizer()
        
        # Adaptive normalization
        self.adaptive_norm = tf.keras.layers.LayerNormalization(name="user_adaptive_norm")
        
        # Global user bias
        if use_bias:
            self.global_user_bias = tf.Variable(
                initial_value=0.0, trainable=True, name="global_user_bias"
            )
    
    def call(self, inputs, training=None):
        """Enhanced forward pass with diversity regularization."""
        age = inputs["age"]
        gender = inputs["gender"]
        income = inputs["income"]
        item_history = inputs["item_history_embeddings"]
        
        # Process demographics
        age_emb = self.age_embedding(age)
        income_emb = self.income_embedding(income)
        gender_emb = self.gender_embedding(gender)
        
        # Combine demographics
        demo_combined = tf.concat([age_emb, income_emb, gender_emb], axis=-1)
        
        # Enhanced history processing
        batch_size = tf.shape(item_history)[0]
        seq_len = tf.shape(item_history)[1]
        
        # Simplified positional encoding - ensure shape compatibility
        positions = tf.range(seq_len, dtype=tf.float32)
        # Create simpler positional encoding
        pos_encoding_scale = tf.cast(tf.range(self.embedding_dim, dtype=tf.float32), tf.float32) / self.embedding_dim
        position_encoding = tf.sin(positions[:, tf.newaxis] * pos_encoding_scale[tf.newaxis, :])
        
        # Ensure correct shape: [seq_len, embedding_dim] -> [batch_size, seq_len, embedding_dim]
        position_encoding = tf.expand_dims(position_encoding, 0)
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
        
        # Add positional encoding with shape check
        history_with_pos = item_history + position_encoding
        
        # Create attention mask - fix shape for MultiHeadAttention
        # MultiHeadAttention expects mask shape: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        history_mask = tf.reduce_sum(tf.abs(item_history), axis=-1) > 0  # [batch_size, seq_len]
        
        # Apply transformer attention
        attended_history = self.history_transformer(
            query=history_with_pos,
            value=history_with_pos,
            key=history_with_pos,
            attention_mask=history_mask,
            training=training
        )
        
        # Attention-based pooling instead of simple mean
        attention_weights = tf.nn.softmax(
            self.history_attention_pooling(attended_history), axis=1
        )
        history_aggregated = tf.reduce_sum(
            attended_history * attention_weights, axis=1
        )
        
        # Combine features
        combined = tf.concat([demo_combined, history_aggregated], axis=-1)
        
        # Pass through dense layers with residual connections
        x = combined
        residual = x
        for i, layer in enumerate(self.dense_layers):
            x = layer(x, training=training)
            # Add residual connection every 4 layers
            if (i + 1) % 4 == 0 and x.shape[-1] == residual.shape[-1]:
                x = x + residual
                residual = x
        
        # Final output
        output = self.output_layer(x)
        
        # Apply diversity regularization if enabled
        if self.use_diversity_reg and training:
            output = self.diversity_regularizer(output)
        
        # Adaptive normalization
        normalized_output = self.adaptive_norm(output)
        
        # Add bias if enabled
        if self.use_bias:
            return normalized_output, self.global_user_bias
        else:
            return normalized_output


class EnhancedTwoTowerModel(tfrs.Model):
    """Enhanced two-tower model with all improvements."""
    
    def __init__(self, 
                 item_tower: EnhancedItemTower,
                 user_tower: EnhancedUserTower,
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0,
                 contrastive_weight: float = 0.3,
                 diversity_weight: float = 0.1):
        super().__init__()
        
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        
        # Adaptive temperature scaling
        self.temperature_similarity = AdaptiveTemperatureScaling()
        
        # Enhanced rating model
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
        
        # Focal loss for imbalanced data
        self.focal_loss = self._focal_loss
        
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss implementation."""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        focal_loss = focal_weight * bce
        
        return tf.reduce_mean(focal_loss)
    
    def call(self, features):
        # Get embeddings
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
    
    def compute_loss(self, features, training=False):
        # Get embeddings and biases
        outputs = self(features)
        user_embeddings = outputs["user_embedding"]
        item_embeddings = outputs["item_embedding"]
        user_bias = outputs["user_bias"]
        item_bias = outputs["item_bias"]
        
        # Rating prediction
        concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
        rating_predictions = self.rating_model(concatenated, training=training)
        
        # Add bias terms
        rating_predictions_with_bias = rating_predictions + user_bias + item_bias
        rating_predictions_with_bias = tf.nn.sigmoid(rating_predictions_with_bias)
        
        # Losses
        rating_loss = self.focal_loss(features["rating"], rating_predictions_with_bias)
        
        # Adaptive temperature-scaled retrieval loss
        scaled_similarities, temperature = self.temperature_similarity(
            user_embeddings, item_embeddings
        )
        retrieval_loss = tf.keras.losses.binary_crossentropy(
            features["rating"], 
            tf.nn.sigmoid(scaled_similarities)
        )
        retrieval_loss = tf.reduce_mean(retrieval_loss)
        
        # Enhanced contrastive loss with hard negatives
        batch_size = tf.shape(user_embeddings)[0]
        positive_similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
        
        # Random negative sampling
        shuffled_indices = tf.random.shuffle(tf.range(batch_size))
        negative_item_embeddings = tf.gather(item_embeddings, shuffled_indices)
        negative_similarities = tf.reduce_sum(user_embeddings * negative_item_embeddings, axis=1)
        
        # Triplet loss with adaptive margin
        margin = 0.5 / temperature  # Adaptive margin based on temperature
        contrastive_loss = tf.reduce_mean(
            tf.maximum(0.0, margin + negative_similarities - positive_similarities)
        )
        
        # Combine losses
        total_loss = (
            self.rating_weight * rating_loss +
            self.retrieval_weight * retrieval_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        # Add regularization losses from diversity regularizers
        if training:
            regularization_losses = tf.add_n(self.losses) if self.losses else 0.0
            total_loss += self.diversity_weight * regularization_losses
        
        return {
            'total_loss': total_loss,
            'rating_loss': rating_loss,
            'retrieval_loss': retrieval_loss,
            'contrastive_loss': contrastive_loss,
            'temperature': temperature,
            'diversity_loss': regularization_losses if training else 0.0
        }


def create_enhanced_model(data_processor, 
                         embedding_dim=128,
                         use_bias=True,
                         use_diversity_reg=True):
    """Factory function to create enhanced two-tower model."""
    
    # Create enhanced towers
    item_tower = EnhancedItemTower(
        item_vocab_size=len(data_processor.item_vocab),
        category_vocab_size=len(data_processor.category_vocab),
        brand_vocab_size=len(data_processor.brand_vocab),
        embedding_dim=embedding_dim,
        use_bias=use_bias,
        use_diversity_reg=use_diversity_reg
    )
    
    user_tower = EnhancedUserTower(
        max_history_length=50,
        embedding_dim=embedding_dim,
        use_bias=use_bias,
        use_diversity_reg=use_diversity_reg
    )
    
    # Create enhanced model
    model = EnhancedTwoTowerModel(
        item_tower=item_tower,
        user_tower=user_tower,
        rating_weight=1.0,
        retrieval_weight=0.5,
        contrastive_weight=0.3,
        diversity_weight=0.1
    )
    
    return model