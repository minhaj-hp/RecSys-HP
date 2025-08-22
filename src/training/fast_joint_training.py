"""
Fast joint training with key optimizations for CPU performance.
"""
import tensorflow as tf
import numpy as np
import pickle
import os
import time
from typing import Dict

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower, TwoTowerModel
from src.preprocessing.data_loader import DataProcessor


class FastJointTrainer:
    """Simplified fast joint training optimized for CPU."""
    
    def __init__(self):
        self.item_tower = None
        self.user_tower = None
        self.model = None
        
        # Optimized hyperparameters for fast training
        self.user_lr = 0.003
        self.item_lr = 0.0003
        self.batch_size = 2048  # Large batch for efficiency
        self.epochs = 20  # Reduced epochs
    
    def load_components(self):
        """Load all required components."""
        print("Loading components...")
        
        # Load data processor
        data_processor = DataProcessor()
        data_processor.load_vocabularies("src/artifacts/vocabularies.pkl")
        
        # Load item tower config
        with open("src/artifacts/item_tower_config.txt", 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['embedding_dim', 'dropout_rate']:
                    config[key] = float(value) if '.' in value else int(value)
                elif key == 'hidden_dims':
                    config[key] = eval(value)
        
        # Build item tower
        self.item_tower = ItemTower(
            item_vocab_size=len(data_processor.item_vocab),
            category_vocab_size=len(data_processor.category_vocab),
            brand_vocab_size=len(data_processor.brand_vocab),
            **config
        )
        
        # Initialize and load weights
        dummy_input = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([0.0])
        }
        _ = self.item_tower(dummy_input)
        self.item_tower.load_weights("src/artifacts/item_tower_weights")
        
        # Build user tower (simplified)
        self.user_tower = UserTower(
            max_history_length=50,
            embedding_dim=64,
            hidden_dims=[64],  # Simplified architecture
            dropout_rate=0.1
        )
        
        # Build complete model
        self.model = TwoTowerModel(
            item_tower=self.item_tower,
            user_tower=self.user_tower,
            rating_weight=1.0,
            retrieval_weight=0.2  # Reduced for faster training
        )
        
        print("Components loaded successfully")
    
    def create_fast_dataset(self, features: Dict, is_training: bool = True):
        """Create optimized dataset pipeline."""
        dataset = tf.data.Dataset.from_tensor_slices(features)
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(2)  # Conservative prefetch for CPU
        
        return dataset
    
    def train_fast(self, training_features: Dict, validation_features: Dict):
        """Fast training loop with key optimizations."""
        
        print(f"Starting fast training: {self.epochs} epochs, batch size {self.batch_size}")
        
        # Setup datasets
        steps_per_epoch = len(training_features['rating']) // self.batch_size
        val_steps = len(validation_features['rating']) // self.batch_size
        
        train_ds = self.create_fast_dataset(training_features, is_training=True)
        val_ds = self.create_fast_dataset(validation_features, is_training=False)
        
        # Note: Age and income are now categorical - no normalization needed
        
        # Setup optimizers
        user_optimizer = tf.keras.optimizers.Adam(learning_rate=self.user_lr)
        item_optimizer = tf.keras.optimizers.Adam(learning_rate=self.item_lr)
        
        # Training loop
        train_iter = iter(train_ds)
        val_iter = iter(val_ds)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Progressive unfreezing - simple strategy
            train_item = epoch >= (self.epochs // 4)  # Unfreeze after 25%
            
            print(f"Epoch {epoch+1}/{self.epochs} - Item training: {'ON' if train_item else 'OFF'}")
            
            # Training
            train_losses = []
            for step in range(steps_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_ds)
                    batch = next(train_iter)
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    user_emb = self.user_tower(batch, training=True)
                    item_emb = self.item_tower(batch, training=True)
                    
                    # Rating prediction
                    concat_emb = tf.concat([user_emb, item_emb], axis=-1)
                    rating_pred = self.model.rating_model(concat_emb, training=True)
                    
                    # Simple loss calculation
                    rating_loss = tf.keras.losses.binary_crossentropy(
                        batch["rating"], tf.squeeze(rating_pred)
                    )
                    rating_loss = tf.reduce_mean(rating_loss)
                    
                    # Simplified retrieval loss
                    similarity = tf.reduce_sum(user_emb * item_emb, axis=1)
                    retrieval_loss = tf.keras.losses.binary_crossentropy(
                        batch["rating"], tf.nn.sigmoid(similarity)
                    )
                    retrieval_loss = tf.reduce_mean(retrieval_loss)
                    
                    total_loss = rating_loss + 0.2 * retrieval_loss
                
                # Gradient computation and application
                if train_item:
                    # Train both towers
                    user_vars = self.user_tower.trainable_variables + self.model.rating_model.trainable_variables
                    item_vars = self.item_tower.trainable_variables
                    all_vars = user_vars + item_vars
                    
                    grads = tape.gradient(total_loss, all_vars)
                    user_grads = grads[:len(user_vars)]
                    item_grads = grads[len(user_vars):]
                    
                    user_optimizer.apply_gradients(zip(user_grads, user_vars))
                    item_optimizer.apply_gradients(zip(item_grads, item_vars))
                else:
                    # Train only user tower
                    user_vars = self.user_tower.trainable_variables + self.model.rating_model.trainable_variables
                    grads = tape.gradient(total_loss, user_vars)
                    user_optimizer.apply_gradients(zip(grads, user_vars))
                
                train_losses.append(total_loss.numpy())
            
            # Validation
            val_losses = []
            for step in range(val_steps):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_ds)
                    batch = next(val_iter)
                
                user_emb = self.user_tower(batch, training=False)
                item_emb = self.item_tower(batch, training=False)
                
                concat_emb = tf.concat([user_emb, item_emb], axis=-1)
                rating_pred = self.model.rating_model(concat_emb, training=False)
                
                rating_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(batch["rating"], tf.squeeze(rating_pred))
                )
                
                similarity = tf.reduce_sum(user_emb * item_emb, axis=1)
                retrieval_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(batch["rating"], tf.nn.sigmoid(similarity))
                )
                
                total_loss = rating_loss + 0.2 * retrieval_loss
                val_losses.append(total_loss.numpy())
            
            # Calculate averages
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            epoch_time = time.time() - epoch_start
            
            print(f"Time: {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model("_best")
        
        print("Fast training completed!")
    
    def save_model(self, suffix=""):
        """Save trained model."""
        save_path = "src/artifacts/"
        
        self.user_tower.save_weights(f"{save_path}/user_tower_weights{suffix}")
        self.item_tower.save_weights(f"{save_path}/item_tower_weights_finetuned{suffix}")
        self.model.rating_model.save_weights(f"{save_path}/rating_model_weights{suffix}")
        
        if not suffix:
            print("Model saved successfully")


def main():
    """Main function for fast joint training."""
    
    print("=== Fast Joint Training ===")
    
    # Initialize trainer
    trainer = FastJointTrainer()
    trainer.load_components()
    
    # Load training data
    print("Loading training data...")
    with open("src/artifacts/training_features.pkl", 'rb') as f:
        training_features = pickle.load(f)
    
    with open("src/artifacts/validation_features.pkl", 'rb') as f:
        validation_features = pickle.load(f)
    
    print(f"Training samples: {len(training_features['rating']):,}")
    print(f"Validation samples: {len(validation_features['rating']):,}")
    
    # Start training
    start_time = time.time()
    trainer.train_fast(training_features, validation_features)
    
    total_time = time.time() - start_time
    trainer.save_model()
    
    print(f"\\nTraining completed in {total_time:.1f} seconds!")
    print(f"Average time per epoch: {total_time/trainer.epochs:.1f}s")


if __name__ == "__main__":
    main()