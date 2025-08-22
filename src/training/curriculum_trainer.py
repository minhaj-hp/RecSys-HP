#!/usr/bin/env python3
"""
Curriculum learning trainer for the improved two-tower model.
Implements progressive difficulty training for better convergence.
"""

import tensorflow as tf
import numpy as np
import pickle
import os
import time
from typing import Dict, List, Tuple

from src.models.improved_two_tower import create_improved_model
from src.preprocessing.data_loader import DataProcessor


class CurriculumTrainer:
    """Trainer with curriculum learning for improved two-tower model."""
    
    def __init__(self,
                 embedding_dim: int = 128,
                 learning_rate: float = 0.001,
                 use_focal_loss: bool = True,
                 curriculum_stages: int = 3):
        
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        self.curriculum_stages = curriculum_stages
        
        self.data_processor = None
        self.model = None
        
    def load_data_processor(self, artifacts_path: str = "src/artifacts/"):
        """Load data processor with vocabularies."""
        self.data_processor = DataProcessor()
        self.data_processor.load_vocabularies(f"{artifacts_path}/vocabularies.pkl")
        print("Data processor loaded successfully")
        
    def create_model(self):
        """Create improved two-tower model."""
        if self.data_processor is None:
            raise ValueError("Data processor must be loaded first")
            
        self.model = create_improved_model(
            data_processor=self.data_processor,
            embedding_dim=self.embedding_dim,
            use_bias=True,
            use_focal_loss=self.use_focal_loss
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        print("Improved two-tower model created successfully")
        
    def _create_curriculum_stages(self, features: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Create curriculum stages based on interaction complexity."""
        
        # Calculate interaction history lengths for curriculum
        history_lengths = []
        for i in range(len(features['age'])):
            hist = features['item_history_embeddings'][i]
            # Count non-zero embeddings
            length = np.sum(np.any(hist != 0, axis=1))
            history_lengths.append(length)
        
        history_lengths = np.array(history_lengths)
        
        # Create stages based on history length percentiles
        stages = []
        
        if self.curriculum_stages == 3:
            # Stage 1: Simple cases (short or no history)
            stage1_mask = history_lengths <= np.percentile(history_lengths, 33)
            
            # Stage 2: Medium complexity (medium history)
            stage2_mask = (history_lengths > np.percentile(history_lengths, 33)) & \
                         (history_lengths <= np.percentile(history_lengths, 67))
            
            # Stage 3: Complex cases (long history)
            stage3_mask = history_lengths > np.percentile(history_lengths, 67)
            
            masks = [stage1_mask, stage2_mask, stage3_mask]
            stage_names = ["Simple (short history)", "Medium (moderate history)", "Complex (long history)"]
            
        else:
            # Flexible number of stages
            percentiles = np.linspace(0, 100, self.curriculum_stages + 1)
            masks = []
            stage_names = []
            
            for i in range(self.curriculum_stages):
                if i == 0:
                    mask = history_lengths <= np.percentile(history_lengths, percentiles[i+1])
                    stage_names.append(f"Stage {i+1} (≤{percentiles[i+1]:.0f}%ile)")
                elif i == self.curriculum_stages - 1:
                    mask = history_lengths > np.percentile(history_lengths, percentiles[i])
                    stage_names.append(f"Stage {i+1} (>{percentiles[i]:.0f}%ile)")
                else:
                    mask = (history_lengths > np.percentile(history_lengths, percentiles[i])) & \
                           (history_lengths <= np.percentile(history_lengths, percentiles[i+1]))
                    stage_names.append(f"Stage {i+1} ({percentiles[i]:.0f}-{percentiles[i+1]:.0f}%ile)")
                
                masks.append(mask)
        
        # Create stage datasets
        for i, (mask, name) in enumerate(zip(masks, stage_names)):
            stage_features = {}
            for key, values in features.items():
                stage_features[key] = values[mask]
            
            print(f"  Stage {i+1} ({name}): {np.sum(mask)} samples")
            stages.append(stage_features)
        
        return stages
    
    def _create_tf_dataset(self, features: Dict[str, np.ndarray], 
                          batch_size: int = 256,
                          shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset from features."""
        
        dataset = tf.data.Dataset.from_tensor_slices(features)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_with_curriculum(self,
                            training_features: Dict[str, np.ndarray],
                            validation_features: Dict[str, np.ndarray],
                            epochs_per_stage: int = 10,
                            batch_size: int = 256) -> Dict:
        """Train model using curriculum learning."""
        
        print(f"🎓 CURRICULUM LEARNING TRAINING")
        print(f"Stages: {self.curriculum_stages} | Epochs per stage: {epochs_per_stage}")
        print("="*70)
        
        # Create curriculum stages
        print("\n📚 Creating curriculum stages...")
        training_stages = self._create_curriculum_stages(training_features)
        
        # Training history
        history = {
            'stage_losses': [],
            'stage_val_losses': [],
            'stage_times': [],
            'total_loss': [],
            'rating_loss': [],
            'retrieval_loss': [],
            'contrastive_loss': [],
            'val_total_loss': [],
            'val_rating_loss': [],
            'val_retrieval_loss': []
        }
        
        # Validation dataset (constant across stages)
        val_dataset = self._create_tf_dataset(validation_features, batch_size, shuffle=False)
        
        total_start_time = time.time()
        
        # Train through curriculum stages
        for stage_idx, stage_features in enumerate(training_stages):
            stage_start_time = time.time()
            
            print(f"\n🎯 STAGE {stage_idx + 1}/{self.curriculum_stages}")
            print(f"Training samples: {len(stage_features['rating'])}")
            
            # Create training dataset for this stage
            train_dataset = self._create_tf_dataset(stage_features, batch_size, shuffle=True)
            
            # Adaptive learning rate (decrease as stages progress)
            stage_lr = self.learning_rate * (0.8 ** stage_idx)
            self.model.optimizer.learning_rate.assign(stage_lr)
            print(f"Learning rate: {stage_lr:.6f}")
            
            # Train on this stage
            stage_history = {'loss': [], 'val_loss': []}
            
            for epoch in range(epochs_per_stage):
                epoch_start = time.time()
                
                # Training step
                train_losses = []
                for batch in train_dataset:
                    with tf.GradientTape() as tape:
                        loss_dict = self.model.compute_loss(batch, training=True)
                        total_loss = loss_dict['total_loss']
                    
                    gradients = tape.gradient(total_loss, self.model.trainable_variables)
                    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    train_losses.append({k: v.numpy() for k, v in loss_dict.items()})
                
                # Average training losses
                avg_train_loss = {}
                for key in train_losses[0].keys():
                    avg_train_loss[key] = np.mean([loss[key] for loss in train_losses])
                
                # Validation step
                val_losses = []
                for batch in val_dataset:
                    loss_dict = self.model.compute_loss(batch, training=False)
                    val_losses.append({k: v.numpy() for k, v in loss_dict.items()})
                
                # Average validation losses
                avg_val_loss = {}
                for key in val_losses[0].keys():
                    avg_val_loss[key] = np.mean([loss[key] for loss in val_losses])
                
                # Record epoch results
                stage_history['loss'].append(avg_train_loss['total_loss'])
                stage_history['val_loss'].append(avg_val_loss['total_loss'])
                
                # Add to overall history
                for key in ['total_loss', 'rating_loss', 'retrieval_loss', 'contrastive_loss']:
                    history[key].append(avg_train_loss[key])
                    history[f'val_{key}'].append(avg_val_loss[key])
                
                epoch_time = time.time() - epoch_start
                print(f"  Epoch {epoch+1:2d}/{epochs_per_stage} | "
                      f"Loss: {avg_train_loss['total_loss']:.4f} | "
                      f"Val: {avg_val_loss['total_loss']:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            
            stage_time = time.time() - stage_start_time
            
            # Record stage results
            history['stage_losses'].append(stage_history['loss'])
            history['stage_val_losses'].append(stage_history['val_loss'])
            history['stage_times'].append(stage_time)
            
            print(f"✅ Stage {stage_idx + 1} completed in {stage_time:.1f}s")
            
            # Save intermediate model after each stage
            self.save_model(f"src/artifacts/", suffix=f"_stage_{stage_idx + 1}")
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎓 CURRICULUM TRAINING COMPLETED!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per stage: {np.mean(history['stage_times']):.1f}s")
        
        return history
    
    def save_model(self, save_path: str = "src/artifacts/", suffix: str = ""):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        self.model.user_tower.save_weights(f"{save_path}/improved_user_tower_weights{suffix}")
        self.model.item_tower.save_weights(f"{save_path}/improved_item_tower_weights{suffix}")
        self.model.rating_model.save_weights(f"{save_path}/improved_rating_model_weights{suffix}")
        
        # Save temperature parameter
        temp_value = self.model.temperature_similarity.temperature.numpy()
        with open(f"{save_path}/temperature_value{suffix}.txt", 'w') as f:
            f.write(str(temp_value))
        
        # Save configuration
        config = {
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'use_focal_loss': self.use_focal_loss,
            'curriculum_stages': self.curriculum_stages
        }
        
        with open(f"{save_path}/curriculum_model_config{suffix}.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        if not suffix:
            print(f"Model saved to {save_path}")


def main():
    """Main function for curriculum training."""
    
    print("🚀 INITIALIZING CURRICULUM TRAINER")
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        embedding_dim=128,
        learning_rate=0.001,
        use_focal_loss=True,
        curriculum_stages=3
    )
    
    # Load data processor
    print("Loading data processor...")
    trainer.load_data_processor()
    
    # Create improved model
    print("Creating improved two-tower model...")
    trainer.create_model()
    
    # Load training data
    print("Loading training data...")
    with open("src/artifacts/training_features.pkl", 'rb') as f:
        training_features = pickle.load(f)
    
    with open("src/artifacts/validation_features.pkl", 'rb') as f:
        validation_features = pickle.load(f)
    
    print(f"Training samples: {len(training_features['rating'])}")
    print(f"Validation samples: {len(validation_features['rating'])}")
    
    # Train with curriculum learning
    start_time = time.time()
    
    history = trainer.train_with_curriculum(
        training_features=training_features,
        validation_features=validation_features,
        epochs_per_stage=15,
        batch_size=512
    )
    
    total_time = time.time() - start_time
    
    # Save final model and history
    print("Saving final model...")
    trainer.save_model()
    
    with open("src/artifacts/curriculum_training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n✅ CURRICULUM TRAINING COMPLETED!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"All improvements implemented successfully!")


if __name__ == "__main__":
    main()