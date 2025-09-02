import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np


class ItemTower(tf.keras.Model):
    """Optimized Item tower for two-tower recommendation architecture.
    
    New architecture with smart dimensionality and feature engineering:
    - product_id: 56D (right-sized for 19K items) 
    - category_id: 16D (efficient for categorical relationships)
    - category_code: 16D (hierarchical category understanding)
    - brand: 16D (prevents overfitting, captures brand identity)
    - price: log(price+1) → z-score → Dense(1→16D) (learns price semantics)
    
    Total input: 120D (vs 385D original) - 3x more efficient!
    """
    
    def __init__(self, 
                 item_vocab_size: int,
                 category_vocab_size: int,
                 category_code_vocab_size: int,
                 brand_vocab_size: int,
                 embedding_dim: int = 128,  # Output embedding dimension
                 hidden_dims: list = [256, 128],  # Internal processing dims
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Smart embedding dimensions for different features
        self.product_embedding_dim = 56   # Main identifier - good capacity
        self.category_embedding_dim = 16  # Categorical - appropriate size
        self.brand_embedding_dim = 16     # Brand identity - efficient
        self.price_embedding_dim = 16     # Learned price semantics
        
        # Embedding layers with optimized dimensions
        self.item_embedding = tf.keras.layers.Embedding(
            item_vocab_size, self.product_embedding_dim, name="item_embedding"
        )
        self.category_embedding = tf.keras.layers.Embedding(
            category_vocab_size, self.category_embedding_dim, name="category_embedding"
        )
        self.category_code_embedding = tf.keras.layers.Embedding(
            category_code_vocab_size, self.category_embedding_dim, name="category_code_embedding"
        )
        self.brand_embedding = tf.keras.layers.Embedding(
            brand_vocab_size, self.brand_embedding_dim, name="brand_embedding"
        )
        
        # Smart price preprocessing pipeline
        self.price_normalization = tf.keras.layers.Normalization(name="price_norm")
        self.price_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", name="price_dense1"),
            tf.keras.layers.Dropout(dropout_rate/2, name="price_dropout"),  
            tf.keras.layers.Dense(self.price_embedding_dim, activation=None, name="price_dense2")
        ], name="price_mlp")
        
        # Calculate total input dimension
        self.total_input_dim = (
            self.product_embedding_dim +      # 56D
            self.category_embedding_dim +     # 16D  
            self.category_embedding_dim +     # 16D (category_code)
            self.brand_embedding_dim +        # 16D
            self.price_embedding_dim          # 16D
        )  # Total: 120D
        
        print(f"📊 ItemTower Input Dimensions:")
        print(f"   Product: {self.product_embedding_dim}D")
        print(f"   Category: {self.category_embedding_dim}D") 
        print(f"   Category Code: {self.category_embedding_dim}D")
        print(f"   Brand: {self.brand_embedding_dim}D")
        print(f"   Price (learned): {self.price_embedding_dim}D")
        print(f"   Total Input: {self.total_input_dim}D → Output: {embedding_dim}D")
        
        # Dense processing layers
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
        
    def _preprocess_price(self, price):
        """Smart price preprocessing: log transform → normalize → learn embeddings."""
        
        # Log transform to handle price skewness (luxury vs budget)
        log_price = tf.math.log1p(price)  # log(price + 1) - handles zeros
        
        # Z-score normalization via the normalization layer
        normalized_price = self.price_normalization(tf.expand_dims(log_price, -1))
        
        # Learn price embeddings (price tiers, quality relationships, etc.)
        price_embedding = self.price_mlp(normalized_price)
        
        return price_embedding
        
    def call(self, inputs, training=None):
        """Forward pass of the optimized item tower."""
        item_id = inputs["product_id"]
        category_id = inputs["category_id"] 
        category_code_id = inputs.get("category_code_id", category_id)  # Fallback if not provided
        brand_id = inputs["brand_id"]
        price = inputs["price"]
        
        # Get embeddings with optimized dimensions
        item_emb = self.item_embedding(item_id)                      # [batch, 56]
        category_emb = self.category_embedding(category_id)          # [batch, 16]
        category_code_emb = self.category_code_embedding(category_code_id)  # [batch, 16]
        brand_emb = self.brand_embedding(brand_id)                   # [batch, 16]
        
        # Smart price preprocessing and embedding
        price_emb = self._preprocess_price(price)                    # [batch, 16]
        
        # Concatenate all features: 56 + 16 + 16 + 16 + 16 = 120D
        combined = tf.concat([
            item_emb,           # Product-specific patterns
            category_emb,       # Category groupings
            category_code_emb,  # Hierarchical category structure  
            brand_emb,          # Brand identity and characteristics
            price_emb           # Learned price semantics and tiers
        ], axis=-1)
        
        # Pass through dense processing layers (120D → hidden_dims → 128D)
        x = combined
        for layer in self.dense_layers:
            x = layer(x, training=training)
            
        # Final output projection
        output = self.output_layer(x)
        
        # L2 normalize for cosine similarity computations
        return tf.nn.l2_normalize(output, axis=-1)

    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'product_embedding_dim': self.product_embedding_dim,
            'category_embedding_dim': self.category_embedding_dim,
            'brand_embedding_dim': self.brand_embedding_dim,
            'price_embedding_dim': self.price_embedding_dim,
            'total_input_dim': self.total_input_dim
        })
        return config


class ItemTowerTrainingModel(tfrs.Model):
    """Training wrapper for optimized item tower with reconstruction loss."""
    
    def __init__(self, item_tower: ItemTower):
        super().__init__()
        self.item_tower = item_tower
        
        # Contrastive learning loss for self-supervised training
        self.contrastive_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        
        # Add regularization for the new architecture
        self.l2_regularizer = tf.keras.regularizers.L2(1e-6)
        
    def call(self, features):
        return self.item_tower(features)
    
    def compute_loss(self, features, training=False):
        """Compute contrastive loss for self-supervised learning."""
        item_embeddings = self(features, training=training)
        
        # Compute pairwise similarities for contrastive learning
        similarities = tf.linalg.matmul(item_embeddings, item_embeddings, transpose_b=True)
        
        # Create positive pairs (diagonal elements)
        batch_size = tf.shape(similarities)[0]
        labels = tf.eye(batch_size)
        
        # Contrastive loss - items should be similar to themselves
        reconstruction_loss = self.contrastive_loss(labels, similarities)
        
        # Add L2 regularization for the optimized embeddings
        regularization_loss = tf.reduce_sum([
            self.l2_regularizer(self.item_tower.item_embedding.embeddings),
            self.l2_regularizer(self.item_tower.category_embedding.embeddings),
            self.l2_regularizer(self.item_tower.category_code_embedding.embeddings),
            self.l2_regularizer(self.item_tower.brand_embedding.embeddings),
        ])
        
        total_loss = reconstruction_loss + regularization_loss
        
        # Log metrics for monitoring
        self.compiled_metrics.update_state(labels, similarities)
        
        return total_loss


# Utility function for creating category code vocabulary from category strings
def create_category_code_vocab(category_codes):
    """Create vocabulary mapping for hierarchical category codes.
    
    Args:
        category_codes: List of category code strings (e.g., ['electronics.audio.headphones'])
        
    Returns:
        vocab_dict: Mapping from category_code to integer ID
    """
    unique_codes = sorted(set(category_codes))
    vocab_dict = {code: idx for idx, code in enumerate(unique_codes)}
    vocab_dict['<UNK>'] = len(vocab_dict)  # Unknown category code
    
    print(f"📚 Created category code vocabulary: {len(vocab_dict)} unique codes")
    print(f"   Examples: {list(unique_codes)[:5]}...")
    
    return vocab_dict


# Helper function to estimate parameter count
def estimate_item_tower_parameters(item_vocab_size, category_vocab_size, 
                                 category_code_vocab_size, brand_vocab_size,
                                 hidden_dims=[256, 128], embedding_dim=128):
    """Estimate parameter count for the new ItemTower architecture."""
    
    # Embedding parameters
    item_emb_params = item_vocab_size * 56
    category_emb_params = category_vocab_size * 16  
    category_code_emb_params = category_code_vocab_size * 16
    brand_emb_params = brand_vocab_size * 16
    
    total_emb_params = item_emb_params + category_emb_params + category_code_emb_params + brand_emb_params
    
    # Price MLP parameters
    price_mlp_params = (1 * 32 + 32) + (32 * 16 + 16)  # Dense layers + biases
    
    # Main dense network parameters  
    input_dim = 120  # 56 + 16 + 16 + 16 + 16
    dense_params = 0
    
    prev_dim = input_dim
    for dim in hidden_dims:
        dense_params += prev_dim * dim + dim  # weights + bias
        prev_dim = dim
    
    # Output layer
    dense_params += prev_dim * embedding_dim + embedding_dim
    
    total_params = total_emb_params + price_mlp_params + dense_params
    
    print(f"📊 Estimated ItemTower Parameters:")
    print(f"   Embeddings: {total_emb_params:,} ({total_emb_params/total_params*100:.1f}%)")
    print(f"   Price MLP: {price_mlp_params:,}")  
    print(f"   Dense Network: {dense_params:,}")
    print(f"   Total: {total_params:,} parameters")
    print(f"   Reduction vs Original (~2.7M): {(1 - total_params/2700000)*100:.1f}% smaller!")
    
    return total_params


if __name__ == "__main__":
    # Test the new architecture
    print("🧪 Testing Optimized ItemTower Architecture")
    print("=" * 50)
    
    # Example vocabulary sizes (from your system)
    estimate_item_tower_parameters(
        item_vocab_size=19095,
        category_vocab_size=238, 
        category_code_vocab_size=500,  # Estimated for hierarchical codes
        brand_vocab_size=1151
    )