#!/usr/bin/env python3
"""
Improved joint training with hard negative mining, curriculum learning, and better optimization.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

from src.models.improved_two_tower import create_improved_model
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset


class HardNegativeSampler:
    """Hard negative sampling strategy for better training."""
    
    def __init__(self, model, item_embeddings, sampling_strategy='mixed'):
        self.model = model
        self.item_embeddings = item_embeddings  # Pre-computed item embeddings
        self.sampling_strategy = sampling_strategy
        
    def sample_hard_negatives(self, user_embeddings, positive_items, k_hard=2, k_random=2):
        """Sample hard negatives based on user-item similarity."""
        batch_size = tf.shape(user_embeddings)[0]
        
        # Compute similarities between users and all items
        similarities = tf.linalg.matmul(user_embeddings, self.item_embeddings, transpose_b=True)
        
        # Mask out positive items
        positive_mask = tf.one_hot(positive_items, depth=tf.shape(self.item_embeddings)[0])
        similarities = similarities - positive_mask * 1e9  # Large negative value
        
        # Get top-k similar items (hard negatives)
        _, hard_negative_indices = tf.nn.top_k(similarities, k=k_hard)
        
        # Sample random negatives
        total_items = tf.shape(self.item_embeddings)[0]
        random_negatives = tf.random.uniform(
            shape=[batch_size, k_random], 
            minval=0, 
            maxval=total_items, 
            dtype=tf.int32
        )
        
        # Combine hard and random negatives
        if self.sampling_strategy == 'hard':
            return hard_negative_indices
        elif self.sampling_strategy == 'random':
            return random_negatives
        else:  # mixed
            return tf.concat([hard_negative_indices, random_negatives], axis=1)


class CurriculumLearningScheduler:
    """Curriculum learning scheduler for progressive difficulty."""
    
    def __init__(self, total_epochs, warmup_epochs=10):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
    def get_difficulty_schedule(self, epoch):
        """Get curriculum parameters for current epoch."""
        if epoch < self.warmup_epochs:
            # Easy phase: more random negatives, lower temperature
            hard_negative_ratio = 0.2
            temperature = 2.0
            negative_samples = 2
        elif epoch < self.total_epochs * 0.6:
            # Medium phase: balanced negatives
            hard_negative_ratio = 0.5
            temperature = 1.0
            negative_samples = 4
        else:
            # Hard phase: more hard negatives, higher temperature
            hard_negative_ratio = 0.8
            temperature = 0.5
            negative_samples = 6
            
        return {
            'hard_negative_ratio': hard_negative_ratio,
            'temperature': temperature,
            'negative_samples': negative_samples
        }


class ImprovedJointTrainer:
    """Enhanced joint trainer with advanced techniques."""
    
    def __init__(self,
                 embedding_dim: int = 128,
                 learning_rate: float = 0.001,
                 use_mixed_precision: bool = True,
                 use_curriculum_learning: bool = True,
                 use_hard_negatives: bool = True):
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.use_curriculum_learning = use_curriculum_learning
        self.use_hard_negatives = use_hard_negatives
        
        # Enable mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        self.model = None
        self.data_processor = None
        self.curriculum_scheduler = None
        self.hard_negative_sampler = None
        
    def setup_model(self, data_processor: DataProcessor):
        """Setup the improved model."""
        self.data_processor = data_processor
        
        # Create improved model
        self.model = create_improved_model(
            data_processor=data_processor,
            embedding_dim=self.embedding_dim,
            use_bias=True,
            use_focal_loss=True
        )
        
        print(f"Created improved two-tower model with {self.embedding_dim}D embeddings")
        
    def setup_curriculum_learning(self, total_epochs: int):
        """Setup curriculum learning scheduler."""
        if self.use_curriculum_learning:
            self.curriculum_scheduler = CurriculumLearningScheduler(
                total_epochs=total_epochs,
                warmup_epochs=max(5, total_epochs // 10)
            )
            print("Curriculum learning enabled")
    
    def setup_hard_negative_sampling(self, item_features: Dict[str, np.ndarray]):
        """Setup hard negative sampling."""
        if self.use_hard_negatives:
            # Pre-compute item embeddings for efficient hard negative sampling
            print("Pre-computing item embeddings for hard negative sampling...")
            
            # Create a dummy batch to get item embeddings
            batch_size = 1000
            total_items = len(item_features['product_id'])
            
            item_embeddings_list = []
            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)
                batch_features = {
                    key: tf.constant(value[i:end_idx]) 
                    for key, value in item_features.items()
                }
                
                item_emb_output = self.model.item_tower(batch_features, training=False)
                if isinstance(item_emb_output, tuple):
                    item_emb = item_emb_output[0]  # Get embeddings, ignore bias
                else:
                    item_emb = item_emb_output
                    
                item_embeddings_list.append(item_emb.numpy())
            
            item_embeddings = np.vstack(item_embeddings_list)
            
            self.hard_negative_sampler = HardNegativeSampler(
                model=self.model,
                item_embeddings=tf.constant(item_embeddings, dtype=tf.float32),
                sampling_strategy='mixed'
            )
            print(f"Hard negative sampling enabled with {len(item_embeddings)} items")
    
    def create_advanced_training_dataset(self, 
                                       features: Dict[str, np.ndarray],
                                       batch_size: int = 256,
                                       epoch: int = 0) -> tf.data.Dataset:
        """Create training dataset with curriculum learning and hard negatives."""
        
        # Get curriculum parameters
        if self.curriculum_scheduler:
            curriculum_params = self.curriculum_scheduler.get_difficulty_schedule(epoch)
            print(f"Epoch {epoch}: {curriculum_params}")
        else:
            curriculum_params = {
                'hard_negative_ratio': 0.5,
                'temperature': 1.0,
                'negative_samples': 4
            }
        
        # Filter data based on curriculum (start with easier examples)
        if epoch < 5:  # Warmup epochs - use only high-confidence positive examples
            positive_mask = features['rating'] == 1.0
            if np.sum(positive_mask) > 0:
                # Sample subset of positives and all negatives
                positive_indices = np.where(positive_mask)[0]
                negative_indices = np.where(features['rating'] == 0.0)[0]
                
                # Sample subset for easier learning
                n_positive_samples = min(len(positive_indices), len(negative_indices))
                selected_positive = np.random.choice(
                    positive_indices, size=n_positive_samples, replace=False
                )
                selected_negative = np.random.choice(
                    negative_indices, size=n_positive_samples, replace=False
                )
                
                selected_indices = np.concatenate([selected_positive, selected_negative])
                np.random.shuffle(selected_indices)
                
                # Filter features
                filtered_features = {
                    key: value[selected_indices] for key, value in features.items()
                }
            else:
                filtered_features = features
        else:
            filtered_features = features
        
        # Create dataset
        dataset = create_tf_dataset(filtered_features, batch_size, shuffle=True)
        
        return dataset
    
    def compile_model(self):
        """Compile model with advanced optimizer."""
        # Use AdamW with learning rate scheduling
        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.01
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Enable mixed precision optimizer if needed
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        self.optimizer = optimizer
        print(f"Model compiled with AdamW optimizer (lr={self.learning_rate})")
    
    @tf.function
    def train_step(self, features):
        """Optimized training step with gradient scaling."""
        with tf.GradientTape() as tape:
            # Forward pass
            loss_dict = self.model.compute_loss(features, training=True)
            total_loss = loss_dict['total_loss']
            
            # Scale loss for mixed precision
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss
        
        # Compute gradients
        if self.use_mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss_dict
    
    def evaluate_model(self, validation_dataset):
        """Evaluate model on validation set."""
        total_losses = defaultdict(list)
        
        for batch in validation_dataset:
            loss_dict = self.model.compute_loss(batch, training=False)
            for key, value in loss_dict.items():
                total_losses[key].append(float(value))
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in total_losses.items()}
        return avg_losses
    
    def train(self,
              training_features: Dict[str, np.ndarray],
              validation_features: Dict[str, np.ndarray],
              epochs: int = 50,
              batch_size: int = 256,
              save_path: str = "src/artifacts/") -> Dict:
        """Enhanced training loop with all improvements."""
        
        print(f"Starting improved training for {epochs} epochs...")
        
        # Setup components
        self.setup_curriculum_learning(epochs)
        self.compile_model()
        
        # Create validation dataset
        validation_dataset = create_tf_dataset(validation_features, batch_size, shuffle=False)
        
        # Training history
        history = defaultdict(list)
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Create training dataset for this epoch (curriculum learning)
            training_dataset = self.create_advanced_training_dataset(
                training_features, batch_size, epoch
            )
            
            # Training
            epoch_losses = defaultdict(list)
            num_batches = 0
            
            for batch in training_dataset:
                loss_dict = self.train_step(batch)
                
                for key, value in loss_dict.items():
                    epoch_losses[key].append(float(value))
                num_batches += 1
            
            # Average training losses
            avg_train_losses = {
                key: np.mean(values) for key, values in epoch_losses.items()
            }
            
            # Validation
            avg_val_losses = self.evaluate_model(validation_dataset)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {avg_train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {avg_val_losses['total_loss']:.4f}")
            print(f"  Val Rating Loss: {avg_val_losses['rating_loss']:.4f}")
            print(f"  Val Retrieval Loss: {avg_val_losses['retrieval_loss']:.4f}")
            
            # Save history
            for key, value in avg_train_losses.items():
                history[f'train_{key}'].append(value)
            for key, value in avg_val_losses.items():
                history[f'val_{key}'].append(value)
            
            # Early stopping and model saving
            current_val_loss = avg_val_losses['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                
                # Save best model
                self.save_model(save_path, suffix='_improved_best')
                print(f"  💾 Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model and history
        self.save_model(save_path, suffix='_improved_final')
        self.save_training_history(dict(history), save_path)
        
        print("✅ Improved training completed!")
        return dict(history)
    
    def save_model(self, save_path: str, suffix: str = ''):
        """Save the trained model components."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        self.model.item_tower.save_weights(f"{save_path}/improved_item_tower_weights{suffix}")
        self.model.user_tower.save_weights(f"{save_path}/improved_user_tower_weights{suffix}")
        
        if hasattr(self.model, 'rating_model'):
            self.model.rating_model.save_weights(f"{save_path}/improved_rating_model_weights{suffix}")
        
        # Save configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'use_mixed_precision': self.use_mixed_precision,
            'use_curriculum_learning': self.use_curriculum_learning,
            'use_hard_negatives': self.use_hard_negatives
        }
        
        with open(f"{save_path}/improved_model_config{suffix}.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Model saved to {save_path} with suffix '{suffix}'")
    
    def save_training_history(self, history: Dict, save_path: str):
        """Save training history."""
        with open(f"{save_path}/improved_training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        print(f"Training history saved to {save_path}")


def main():
    """Demo of improved training."""
    print("🚀 IMPROVED TWO-TOWER TRAINING DEMO")
    print("="*60)
    
    # Load data
    print("Loading training data...")
    try:
        with open("src/artifacts/training_features.pkl", 'rb') as f:
            training_features = pickle.load(f)
        with open("src/artifacts/validation_features.pkl", 'rb') as f:
            validation_features = pickle.load(f)
        
        print(f"Loaded {len(training_features['rating'])} training samples")
        print(f"Loaded {len(validation_features['rating'])} validation samples")
    except FileNotFoundError:
        print("❌ Training data not found. Please run data preparation first.")
        return
    
    # Load data processor
    data_processor = DataProcessor()
    data_processor.load_vocabularies("src/artifacts/vocabularies.pkl")
    
    # Create trainer
    trainer = ImprovedJointTrainer(
        embedding_dim=128,
        learning_rate=0.001,
        use_mixed_precision=True,
        use_curriculum_learning=True,
        use_hard_negatives=True
    )
    
    # Setup and train
    trainer.setup_model(data_processor)
    
    # Train model
    history = trainer.train(
        training_features=training_features,
        validation_features=validation_features,
        epochs=30,
        batch_size=256
    )
    
    print("✅ Improved training completed successfully!")


if __name__ == "__main__":
    main()