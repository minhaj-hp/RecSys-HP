import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


class ItemTower(tf.keras.Model):
    """Item tower for two-tower recommendation architecture."""
    
    def __init__(self, 
                 item_vocab_size: int,
                 category_vocab_size: int,
                 brand_vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dims: list = [128, 64],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.item_embedding = tf.keras.layers.Embedding(
            item_vocab_size, embedding_dim, name="item_embedding"
        )
        self.category_embedding = tf.keras.layers.Embedding(
            category_vocab_size, embedding_dim, name="category_embedding"
        )
        self.brand_embedding = tf.keras.layers.Embedding(
            brand_vocab_size, embedding_dim, name="brand_embedding"
        )
        
        # Price normalization
        self.price_normalization = tf.keras.layers.Normalization(name="price_norm")
        
        # Dense layers
        self.dense_layers = []
        for i, dim in enumerate(hidden_dims):
            self.dense_layers.extend([
                tf.keras.layers.Dense(dim, activation="relu", name=f"dense_{i}"),
                tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
            ])
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            embedding_dim, activation=None, name="item_output"
        )
        
    def call(self, inputs, training=None):
        """Forward pass of the item tower."""
        item_id = inputs["product_id"]
        category_id = inputs["category_id"] 
        brand_id = inputs["brand_id"]
        price = inputs["price"]
        
        # Get embeddings
        item_emb = self.item_embedding(item_id)
        category_emb = self.category_embedding(category_id)
        brand_emb = self.brand_embedding(brand_id)
        
        # Normalize price and expand dims
        price_norm = self.price_normalization(tf.expand_dims(price, -1))
        
        # Concatenate all features
        combined = tf.concat([
            item_emb, 
            category_emb, 
            brand_emb, 
            price_norm
        ], axis=-1)
        
        # Pass through dense layers
        x = combined
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        # Final output
        output = self.output_layer(x)
        
        # L2 normalize for similarity computations
        return tf.nn.l2_normalize(output, axis=-1)


class ItemTowerTrainingModel(tfrs.Model):
    """Training wrapper for item tower with reconstruction loss."""
    
    def __init__(self, item_tower: ItemTower):
        super().__init__()
        self.item_tower = item_tower
        
        # Reconstruction task for self-supervised learning
        self.retrieval_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        
    def call(self, features):
        return self.item_tower(features)
    
    def compute_loss(self, features, training=False):
        item_embeddings = self(features)
        
        # Simple contrastive loss for self-supervised learning
        # Compute pairwise similarities
        similarities = tf.linalg.matmul(item_embeddings, item_embeddings, transpose_b=True)
        
        # Create positive pairs (diagonal elements)
        batch_size = tf.shape(similarities)[0]
        labels = tf.eye(batch_size)
        
        # Contrastive loss
        reconstruction_loss = self.retrieval_loss(labels, similarities)
        
        # Return scalar loss for TFX compatibility
        return reconstruction_loss