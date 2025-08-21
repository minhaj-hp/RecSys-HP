import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List
import os

from src.models.item_tower import ItemTower, ItemTowerTrainingModel
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset


class ItemTowerPretrainer:
    """Handles pre-training of the item tower."""
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.item_tower = None
        
    def prepare_data(self, data_processor: DataProcessor):
        """Prepare item data for pre-training."""
        items_df, users_df, interactions_df = data_processor.load_data()
        data_processor.build_vocabularies(items_df, users_df, interactions_df)
        
        # Prepare item features
        item_features = data_processor.prepare_item_features(items_df)
        
        # Create TensorFlow dataset
        dataset = create_tf_dataset(item_features, batch_size=512)
        
        # Initialize price normalization layer
        all_prices = np.array(item_features['price']).reshape(-1, 1)  # Ensure proper shape
        price_normalizer = tf.keras.layers.Normalization()
        price_normalizer.adapt(all_prices)
        
        return dataset, data_processor, price_normalizer
    
    def build_model(self, 
                    item_vocab_size: int,
                    category_vocab_size: int, 
                    brand_vocab_size: int,
                    price_normalizer: tf.keras.layers.Normalization):
        """Build item tower model for pre-training."""
        
        self.item_tower = ItemTower(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            brand_vocab_size=brand_vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        # Set pre-adapted price normalizer
        self.item_tower.price_normalization = price_normalizer
        
        self.model = ItemTowerTrainingModel(self.item_tower)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            run_eagerly=False
        )
        
        return self.model
    
    def train(self, 
              dataset: tf.data.Dataset,
              epochs: int = 50,
              validation_split: float = 0.2):
        """Train the item tower."""
        
        # Split dataset for validation
        total_batches = len(list(dataset))
        train_size = int(total_batches * (1 - validation_split))
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_total_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def generate_item_embeddings(self, 
                                dataset: tf.data.Dataset) -> Dict[int, np.ndarray]:
        """Generate embeddings for all items in the catalog."""
        
        item_embeddings = {}
        
        for batch in dataset:
            embeddings = self.item_tower(batch)
            product_ids = batch['product_id'].numpy()
            
            for i, product_id in enumerate(product_ids):
                item_embeddings[product_id] = embeddings[i].numpy()
        
        print(f"Generated embeddings for {len(item_embeddings)} items")
        return item_embeddings
    
    def save_model(self, save_path: str = "src/artifacts/"):
        """Save the pre-trained item tower."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save the item tower
        self.item_tower.save_weights(f"{save_path}/item_tower_weights")
        
        # Save model architecture
        with open(f"{save_path}/item_tower_config.txt", 'w') as f:
            f.write(f"embedding_dim: {self.embedding_dim}\n")
            f.write(f"hidden_dims: {self.hidden_dims}\n")
            f.write(f"dropout_rate: {self.dropout_rate}\n")
        
        print(f"Item tower saved to {save_path}")
    
    def load_model(self, 
                   load_path: str = "src/artifacts/",
                   item_vocab_size: int = None,
                   category_vocab_size: int = None,
                   brand_vocab_size: int = None):
        """Load pre-trained item tower."""
        
        # Read config
        with open(f"{load_path}/item_tower_config.txt", 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['embedding_dim', 'dropout_rate']:
                    config[key] = float(value) if '.' in value else int(value)
                elif key == 'hidden_dims':
                    config[key] = eval(value)  # Parse list
        
        # Build model architecture
        self.item_tower = ItemTower(
            item_vocab_size=item_vocab_size,
            category_vocab_size=category_vocab_size,
            brand_vocab_size=brand_vocab_size,
            **config
        )
        
        # Load weights (need to build model first)
        dummy_input = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([0.0])
        }
        _ = self.item_tower(dummy_input)  # Build model
        
        self.item_tower.load_weights(f"{load_path}/item_tower_weights")
        print(f"Item tower loaded from {load_path}")


def main():
    """Main function for item tower pre-training."""
    
    # Initialize components
    data_processor = DataProcessor()
    pretrainer = ItemTowerPretrainer(
        embedding_dim=64,
        hidden_dims=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Prepare data
    print("Preparing data...")
    dataset, data_processor, price_normalizer = pretrainer.prepare_data(data_processor)
    
    # Build model
    print("Building model...")
    model = pretrainer.build_model(
        item_vocab_size=len(data_processor.item_vocab),
        category_vocab_size=len(data_processor.category_vocab),
        brand_vocab_size=len(data_processor.brand_vocab),
        price_normalizer=price_normalizer
    )
    
    # Train model
    print("Training item tower...")
    history = pretrainer.train(dataset, epochs=50)
    
    # Generate embeddings
    print("Generating item embeddings...")
    item_embeddings = pretrainer.generate_item_embeddings(dataset)
    
    # Save everything
    print("Saving artifacts...")
    data_processor.save_vocabularies()
    pretrainer.save_model()
    
    # Save embeddings
    os.makedirs("src/artifacts", exist_ok=True)
    np.save("src/artifacts/item_embeddings.npy", item_embeddings)
    
    print("Item tower pre-training completed!")


if __name__ == "__main__":
    main()