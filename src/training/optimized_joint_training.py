import tensorflow as tf
import numpy as np
import pickle
import os
import time
from typing import Dict, List, Tuple

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower, TwoTowerModel
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset


class OptimizedJointTrainer:
    """Optimized joint training with performance enhancements."""
    
    def __init__(self,
                 embedding_dim: int = 64,
                 user_learning_rate: float = 0.001,
                 item_learning_rate: float = 0.0001,
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0,
                 gradient_accumulation_steps: int = 1,
                 use_mixed_precision: bool = False):  # Disabled for CPU training
        
        self.embedding_dim = embedding_dim
        self.user_learning_rate = user_learning_rate
        self.item_learning_rate = item_learning_rate
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        
        # Enable mixed precision for faster training
        if self.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision training enabled")
        
        self.item_tower = None
        self.user_tower = None
        self.model = None
        
        # Precompile TensorFlow functions for speed
        self._compiled_train_step = None
        self._compiled_val_step = None
    
    def load_pre_trained_item_tower(self, artifacts_path: str = "src/artifacts/") -> ItemTower:
        """Load pre-trained item tower with optimizations."""
        data_processor = DataProcessor()
        data_processor.load_vocabularies(f"{artifacts_path}/vocabularies.pkl")
        
        with open(f"{artifacts_path}/item_tower_config.txt", 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['embedding_dim', 'dropout_rate']:
                    config[key] = float(value) if '.' in value else int(value)
                elif key == 'hidden_dims':
                    config[key] = eval(value)
        
        self.item_tower = ItemTower(
            item_vocab_size=len(data_processor.item_vocab),
            category_vocab_size=len(data_processor.category_vocab),
            brand_vocab_size=len(data_processor.brand_vocab),
            **config
        )
        
        dummy_input = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([0.0])
        }
        _ = self.item_tower(dummy_input)
        self.item_tower.load_weights(f"{artifacts_path}/item_tower_weights")
        
        print("Pre-trained item tower loaded successfully")
        return self.item_tower
    
    def build_user_tower(self, max_history_length: int = 50) -> UserTower:
        """Build user tower with optimizations."""
        self.user_tower = UserTower(
            max_history_length=max_history_length,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.1  # Reduced dropout for faster training
        )
        
        print("User tower initialized")
        return self.user_tower
    
    def build_two_tower_model(self) -> TwoTowerModel:
        """Build complete two-tower model."""
        if self.item_tower is None or self.user_tower is None:
            raise ValueError("Both towers must be initialized first")
        
        self.model = TwoTowerModel(
            item_tower=self.item_tower,
            user_tower=self.user_tower,
            rating_weight=self.rating_weight,
            retrieval_weight=self.retrieval_weight
        )
        
        print("Two-tower model built successfully")
        return self.model
    
    def create_optimized_dataset(self, features: Dict[str, np.ndarray], 
                               batch_size: int, 
                               is_training: bool = True) -> tf.data.Dataset:
        """Create optimized dataset pipeline for faster training."""
        
        dataset = tf.data.Dataset.from_tensor_slices(features)
        
        if is_training:
            # Optimized shuffling and prefetching
            dataset = dataset.shuffle(buffer_size=min(10000, len(features['rating'])))
            dataset = dataset.repeat()  # Repeat for multiple epochs
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Optimize for CPU training
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_train_step(self, batch: Dict[str, tf.Tensor], 
                           user_optimizer: tf.keras.optimizers.Optimizer,
                           item_optimizer: tf.keras.optimizers.Optimizer,
                           train_item: bool) -> Dict[str, tf.Tensor]:
        """Optimized training step with tf.function compilation."""
        
        with tf.GradientTape() as tape:
            # Forward pass
            user_embeddings = self.user_tower(batch, training=True)
            item_embeddings = self.item_tower(batch, training=True)
            
            # Concatenate and predict rating
            concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
            rating_predictions = self.model.rating_model(concatenated, training=True)
            
            # Compute losses - fix shape mismatch
            rating_loss = tf.keras.losses.binary_crossentropy(
                tf.expand_dims(batch["rating"], -1), rating_predictions
            )
            rating_loss = tf.reduce_mean(rating_loss)
            
            # Retrieval loss - cosine similarity
            user_norm = tf.nn.l2_normalize(user_embeddings, axis=1)
            item_norm = tf.nn.l2_normalize(item_embeddings, axis=1)
            similarities = tf.reduce_sum(user_norm * item_norm, axis=1)
            
            retrieval_loss = tf.keras.losses.binary_crossentropy(
                batch["rating"], tf.nn.sigmoid(similarities)
            )
            retrieval_loss = tf.reduce_mean(retrieval_loss)
            
            total_loss = (
                self.rating_weight * rating_loss +
                self.retrieval_weight * retrieval_loss
            )
            
            # Handle mixed precision
            if self.use_mixed_precision:
                total_loss = user_optimizer.get_scaled_loss(total_loss)
        
        # Compute gradients
        user_vars = self.user_tower.trainable_variables + self.model.rating_model.trainable_variables
        
        if train_item:
            all_vars = user_vars + self.item_tower.trainable_variables
            gradients = tape.gradient(total_loss, all_vars)
            
            if self.use_mixed_precision:
                gradients = user_optimizer.get_unscaled_gradients(gradients)
            
            # Split gradients
            user_grads = gradients[:len(user_vars)]
            item_grads = gradients[len(user_vars):]
            
            # Apply gradients
            user_optimizer.apply_gradients(zip(user_grads, user_vars))
            item_optimizer.apply_gradients(zip(item_grads, self.item_tower.trainable_variables))
        else:
            gradients = tape.gradient(total_loss, user_vars)
            
            if self.use_mixed_precision:
                gradients = user_optimizer.get_unscaled_gradients(gradients)
            
            user_optimizer.apply_gradients(zip(gradients, user_vars))
        
        # Convert back from scaled loss for logging
        if self.use_mixed_precision:
            total_loss = total_loss / user_optimizer.loss_scale
            rating_loss = rating_loss / user_optimizer.loss_scale
            retrieval_loss = retrieval_loss / user_optimizer.loss_scale
        
        return {
            'total_loss': total_loss,
            'rating_loss': rating_loss,
            'retrieval_loss': retrieval_loss
        }
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_val_step(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Optimized validation step."""
        
        user_embeddings = self.user_tower(batch, training=False)
        item_embeddings = self.item_tower(batch, training=False)
        
        concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
        rating_predictions = self.model.rating_model(concatenated, training=False)
        
        rating_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.expand_dims(batch["rating"], -1), rating_predictions)
        )
        
        # Retrieval loss
        user_norm = tf.nn.l2_normalize(user_embeddings, axis=1)
        item_norm = tf.nn.l2_normalize(item_embeddings, axis=1)
        similarities = tf.reduce_sum(user_norm * item_norm, axis=1)
        
        retrieval_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(batch["rating"], tf.nn.sigmoid(similarities))
        )
        
        total_loss = self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss
        
        return {
            'total_loss': total_loss,
            'rating_loss': rating_loss,
            'retrieval_loss': retrieval_loss
        }
    
    def train(self,
              training_features: Dict[str, np.ndarray],
              validation_features: Dict[str, np.ndarray],
              epochs: int = 50,  # Reduced default epochs
              batch_size: int = 512) -> Dict:  # Larger batch size for efficiency
        """Optimized training loop."""
        
        print(f"Starting optimized joint training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        
        # Create optimized datasets
        steps_per_epoch = len(training_features['rating']) // batch_size
        val_steps = len(validation_features['rating']) // batch_size
        
        train_dataset = self.create_optimized_dataset(training_features, batch_size, is_training=True)
        val_dataset = self.create_optimized_dataset(validation_features, batch_size, is_training=False)
        
        # Note: Age and income are now categorical - no normalization needed
        
        # Setup optimizers with mixed precision
        if self.use_mixed_precision:
            user_optimizer = tf.keras.optimizers.Adam(learning_rate=self.user_learning_rate)
            user_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(user_optimizer)
            item_optimizer = tf.keras.optimizers.Adam(learning_rate=self.item_learning_rate)
            item_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(item_optimizer)
        else:
            user_optimizer = tf.keras.optimizers.Adam(learning_rate=self.user_learning_rate)
            item_optimizer = tf.keras.optimizers.Adam(learning_rate=self.item_learning_rate)
        
        # Training history
        history = {
            'total_loss': [], 'rating_loss': [], 'retrieval_loss': [],
            'val_total_loss': [], 'val_rating_loss': [], 'val_retrieval_loss': [],
            'epoch_times': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Reduced patience for faster training
        
        train_iter = iter(train_dataset)
        val_iter = iter(val_dataset)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\\nEpoch {epoch + 1}/{epochs}")
            
            # Determine training strategy
            freeze_threshold = int(0.2 * epochs)  # Reduced freeze period
            train_item = epoch >= freeze_threshold
            
            print(f"Training: User=✓, Item={'✓' if train_item else '✗'}")
            
            # Training loop
            epoch_losses = {'total_loss': [], 'rating_loss': [], 'retrieval_loss': []}
            
            for step in range(steps_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataset)
                    batch = next(train_iter)
                
                losses = self.optimized_train_step(batch, user_optimizer, item_optimizer, train_item)
                
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key])
            
            # Calculate training averages
            avg_train_losses = {k: tf.reduce_mean(v).numpy() for k, v in epoch_losses.items()}
            
            # Validation loop
            val_losses = {'total_loss': [], 'rating_loss': [], 'retrieval_loss': []}
            
            for step in range(val_steps):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataset)
                    batch = next(val_iter)
                
                losses = self.optimized_val_step(batch)
                
                for key in val_losses:
                    val_losses[key].append(losses[key])
            
            avg_val_losses = {k: tf.reduce_mean(v).numpy() for k, v in val_losses.items()}
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            
            for key in ['total_loss', 'rating_loss', 'retrieval_loss']:
                history[key].append(avg_train_losses[key])
                history[f'val_{key}'].append(avg_val_losses[key])
            
            # Print progress
            print(f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {avg_train_losses['total_loss']:.4f} | "
                  f"Val Loss: {avg_val_losses['total_loss']:.4f}")
            
            # Early stopping with model saving
            if avg_val_losses['total_loss'] < best_val_loss:
                best_val_loss = avg_val_losses['total_loss']
                patience_counter = 0
                self.save_model("src/artifacts/", suffix="_best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        avg_epoch_time = np.mean(history['epoch_times'])
        print(f"\\nTraining completed!")
        print(f"Average epoch time: {avg_epoch_time:.1f}s")
        print(f"Total training time: {sum(history['epoch_times']):.1f}s")
        
        return history
    
    def save_model(self, save_path: str = "src/artifacts/", suffix: str = ""):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        
        self.user_tower.save_weights(f"{save_path}/user_tower_weights{suffix}")
        self.item_tower.save_weights(f"{save_path}/item_tower_weights_finetuned{suffix}")
        self.model.rating_model.save_weights(f"{save_path}/rating_model_weights{suffix}")
        
        config = {
            'embedding_dim': self.embedding_dim,
            'user_learning_rate': self.user_learning_rate,
            'item_learning_rate': self.item_learning_rate,
            'rating_weight': self.rating_weight,
            'retrieval_weight': self.retrieval_weight,
            'use_mixed_precision': self.use_mixed_precision
        }
        
        with open(f"{save_path}/optimized_joint_model_config{suffix}.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\\n")
        
        if not suffix:
            print(f"Optimized model saved to {save_path}")


def main():
    """Main function for optimized joint training."""
    
    print("Initializing optimized joint trainer...")
    trainer = OptimizedJointTrainer(
        embedding_dim=64,
        user_learning_rate=0.002,  # Slightly higher for faster convergence
        item_learning_rate=0.0002,
        rating_weight=1.0,
        retrieval_weight=0.3,  # Reduced for faster training
        use_mixed_precision=False  # Disabled for CPU
    )
    
    # Load components
    print("Loading pre-trained item tower...")
    trainer.load_pre_trained_item_tower()
    
    print("Building user tower...")
    trainer.build_user_tower(max_history_length=50)
    
    print("Building two-tower model...")
    trainer.build_two_tower_model()
    
    # Load training data
    print("Loading training data...")
    with open("src/artifacts/training_features.pkl", 'rb') as f:
        training_features = pickle.load(f)
    
    with open("src/artifacts/validation_features.pkl", 'rb') as f:
        validation_features = pickle.load(f)
    
    print(f"Training samples: {len(training_features['rating'])}")
    print(f"Validation samples: {len(validation_features['rating'])}")
    
    # Train with optimizations
    print("Starting optimized training...")
    start_time = time.time()
    
    history = trainer.train(
        training_features=training_features,
        validation_features=validation_features,
        epochs=30,  # Reduced epochs for faster training
        batch_size=1024  # Larger batch size for better GPU utilization
    )
    
    total_time = time.time() - start_time
    
    # Save final model and history
    print("Saving final model...")
    trainer.save_model()
    
    with open("src/artifacts/optimized_training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\\nOptimized joint training completed!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Average time per epoch: {total_time/len(history['epoch_times']):.1f}s")


if __name__ == "__main__":
    main()