import tensorflow as tf
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower, TwoTowerModel
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset


class JointTrainer:
    """Handles joint training of user and item towers."""
    
    def __init__(self,
                 embedding_dim: int = 64,
                 user_learning_rate: float = 0.001,
                 item_learning_rate: float = 0.0001,  # Lower LR for pre-trained item tower
                 rating_weight: float = 1.0,
                 retrieval_weight: float = 1.0):
        
        self.embedding_dim = embedding_dim
        self.user_learning_rate = user_learning_rate
        self.item_learning_rate = item_learning_rate
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        
        self.item_tower = None
        self.user_tower = None
        self.model = None
    
    def load_pre_trained_item_tower(self, 
                                  artifacts_path: str = "src/artifacts/") -> ItemTower:
        """Load pre-trained item tower."""
        
        # Load vocabularies
        data_processor = DataProcessor()
        data_processor.load_vocabularies(f"{artifacts_path}/vocabularies.pkl")
        
        # Read item tower config
        with open(f"{artifacts_path}/item_tower_config.txt", 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['embedding_dim', 'dropout_rate']:
                    config[key] = float(value) if '.' in value else int(value)
                elif key == 'hidden_dims':
                    config[key] = eval(value)
        
        # Initialize item tower
        self.item_tower = ItemTower(
            item_vocab_size=len(data_processor.item_vocab),
            category_vocab_size=len(data_processor.category_vocab),
            brand_vocab_size=len(data_processor.brand_vocab),
            **config
        )
        
        # Load weights
        dummy_input = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([0.0])
        }
        _ = self.item_tower(dummy_input)  # Build model
        self.item_tower.load_weights(f"{artifacts_path}/item_tower_weights")
        
        print("Pre-trained item tower loaded successfully")
        return self.item_tower
    
    def build_user_tower(self, max_history_length: int = 50) -> UserTower:
        """Build user tower."""
        
        self.user_tower = UserTower(
            max_history_length=max_history_length,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64],
            dropout_rate=0.2
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
    
    def setup_optimizers(self) -> tf.keras.optimizers.Optimizer:
        """Setup optimizers with different learning rates for each tower."""
        
        # Create separate optimizers
        item_optimizer = tf.keras.optimizers.Adam(learning_rate=self.item_learning_rate)
        user_optimizer = tf.keras.optimizers.Adam(learning_rate=self.user_learning_rate)
        
        # We'll use a custom training loop for different learning rates
        # For now, use the user optimizer as primary
        return user_optimizer
    
    def gradual_unfreezing_schedule(self, epoch: int, total_epochs: int) -> Tuple[bool, bool]:
        """Determine whether to train each tower based on unfreezing schedule."""
        
        # First 25% of epochs: freeze item tower, train user tower only
        freeze_threshold = int(0.25 * total_epochs)
        
        train_user = True  # Always train user tower
        train_item = epoch >= freeze_threshold  # Unfreeze item tower after threshold
        
        return train_user, train_item
    
    def custom_training_step(self, 
                           features: Dict[str, tf.Tensor],
                           epoch: int,
                           total_epochs: int) -> Dict[str, tf.Tensor]:
        """Custom training step with different learning rates and gradual unfreezing."""
        
        train_user, train_item = self.gradual_unfreezing_schedule(epoch, total_epochs)
        
        # Create optimizers
        item_optimizer = tf.keras.optimizers.Adam(learning_rate=self.item_learning_rate)
        user_optimizer = tf.keras.optimizers.Adam(learning_rate=self.user_learning_rate)
        
        with tf.GradientTape() as tape:
            # Forward pass
            user_embeddings = self.user_tower(features, training=True)
            item_embeddings = self.item_tower(features, training=True)
            
            # Compute losses
            concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
            rating_predictions = self.model.rating_model(concatenated, training=True)
            
            rating_loss = self.model.rating_task(
                labels=features["rating"],
                predictions=rating_predictions
            )
            
            # Retrieval loss - dot product similarity
            similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
            retrieval_loss = self.model.retrieval_loss(features["rating"], tf.nn.sigmoid(similarities))
            
            total_loss = (
                self.rating_weight * rating_loss +
                self.retrieval_weight * retrieval_loss
            )
        
        # Compute gradients
        trainable_vars = []
        
        if train_user:
            trainable_vars.extend(self.user_tower.trainable_variables)
            trainable_vars.extend(self.model.rating_model.trainable_variables)
        
        if train_item:
            trainable_vars.extend(self.item_tower.trainable_variables)
        
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Apply gradients with appropriate optimizer
        if train_user and train_item:
            # Split gradients for different optimizers
            user_vars = self.user_tower.trainable_variables + self.model.rating_model.trainable_variables
            item_vars = self.item_tower.trainable_variables
            
            user_grads = gradients[:len(user_vars)]
            item_grads = gradients[len(user_vars):]
            
            user_optimizer.apply_gradients(zip(user_grads, user_vars))
            item_optimizer.apply_gradients(zip(item_grads, item_vars))
            
        elif train_user:
            user_optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            'total_loss': total_loss,
            'rating_loss': rating_loss,
            'retrieval_loss': retrieval_loss,
            'train_user': train_user,
            'train_item': train_item
        }
    
    def train(self,
              training_features: Dict[str, np.ndarray],
              validation_features: Dict[str, np.ndarray],
              epochs: int = 100,
              batch_size: int = 256) -> Dict:
        """Train the two-tower model."""
        
        # Create datasets
        train_dataset = create_tf_dataset(training_features, batch_size)
        val_dataset = create_tf_dataset(validation_features, batch_size)
        
        # Initialize age and income normalizers
        self.user_tower.age_normalization.adapt(np.array(training_features['age']).reshape(-1, 1))
        self.user_tower.income_normalization.adapt(np.array(training_features['income']).reshape(-1, 1))
        
        # Training history
        history = {
            'total_loss': [],
            'rating_loss': [],
            'retrieval_loss': [],
            'val_total_loss': [],
            'val_rating_loss': [],
            'val_retrieval_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f"Starting joint training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            epoch_losses = {'total_loss': [], 'rating_loss': [], 'retrieval_loss': []}
            train_user, train_item = self.gradual_unfreezing_schedule(epoch, epochs)
            
            print(f"Training: User={'✓' if train_user else '✗'}, Item={'✓' if train_item else '✗'}")
            
            for batch in train_dataset:
                losses = self.custom_training_step(batch, epoch, epochs)
                
                epoch_losses['total_loss'].append(losses['total_loss'])
                epoch_losses['rating_loss'].append(losses['rating_loss'])
                epoch_losses['retrieval_loss'].append(losses['retrieval_loss'])
            
            # Calculate average training losses
            avg_train_losses = {k: tf.reduce_mean(v).numpy() for k, v in epoch_losses.items()}
            
            # Validation
            val_losses = {'total_loss': [], 'rating_loss': [], 'retrieval_loss': []}
            
            for batch in val_dataset:
                user_embeddings = self.user_tower(batch, training=False)
                item_embeddings = self.item_tower(batch, training=False)
                
                concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
                rating_predictions = self.model.rating_model(concatenated, training=False)
                
                rating_loss = self.model.rating_task(
                    labels=batch["rating"],
                    predictions=rating_predictions
                )
                
                # Retrieval loss
                similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
                retrieval_loss = self.model.retrieval_loss(batch["rating"], tf.nn.sigmoid(similarities))
                
                total_loss = self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss
                
                val_losses['total_loss'].append(total_loss)
                val_losses['rating_loss'].append(rating_loss)
                val_losses['retrieval_loss'].append(retrieval_loss)
            
            avg_val_losses = {k: tf.reduce_mean(v).numpy() for k, v in val_losses.items()}
            
            # Update history
            history['total_loss'].append(avg_train_losses['total_loss'])
            history['rating_loss'].append(avg_train_losses['rating_loss'])
            history['retrieval_loss'].append(avg_train_losses['retrieval_loss'])
            history['val_total_loss'].append(avg_val_losses['total_loss'])
            history['val_rating_loss'].append(avg_val_losses['rating_loss'])
            history['val_retrieval_loss'].append(avg_val_losses['retrieval_loss'])
            
            # Print losses
            print(f"Train - Total: {avg_train_losses['total_loss']:.4f}, "
                  f"Rating: {avg_train_losses['rating_loss']:.4f}, "
                  f"Retrieval: {avg_train_losses['retrieval_loss']:.4f}")
            print(f"Val   - Total: {avg_val_losses['total_loss']:.4f}, "
                  f"Rating: {avg_val_losses['rating_loss']:.4f}, "
                  f"Retrieval: {avg_val_losses['retrieval_loss']:.4f}")
            
            # Early stopping
            if avg_val_losses['total_loss'] < best_val_loss:
                best_val_loss = avg_val_losses['total_loss']
                patience_counter = 0
                # Save best model
                self.save_model("src/artifacts/", suffix="_best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        print("Joint training completed!")
        return history
    
    def save_model(self, save_path: str = "src/artifacts/", suffix: str = ""):
        """Save the trained two-tower model."""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save user tower
        self.user_tower.save_weights(f"{save_path}/user_tower_weights{suffix}")
        
        # Save updated item tower
        self.item_tower.save_weights(f"{save_path}/item_tower_weights_finetuned{suffix}")
        
        # Save rating model
        self.model.rating_model.save_weights(f"{save_path}/rating_model_weights{suffix}")
        
        # Save model config
        config = {
            'embedding_dim': self.embedding_dim,
            'user_learning_rate': self.user_learning_rate,
            'item_learning_rate': self.item_learning_rate,
            'rating_weight': self.rating_weight,
            'retrieval_weight': self.retrieval_weight
        }
        
        with open(f"{save_path}/joint_model_config{suffix}.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Two-tower model saved to {save_path}")


def main():
    """Main function for joint training."""
    
    # Initialize trainer
    trainer = JointTrainer(
        embedding_dim=64,
        user_learning_rate=0.001,
        item_learning_rate=0.0001,
        rating_weight=1.0,
        retrieval_weight=0.5
    )
    
    # Load pre-trained item tower
    print("Loading pre-trained item tower...")
    trainer.load_pre_trained_item_tower()
    
    # Build user tower
    print("Building user tower...")
    trainer.build_user_tower(max_history_length=50)
    
    # Build complete model
    print("Building two-tower model...")
    trainer.build_two_tower_model()
    
    # Load training data
    print("Loading training data...")
    with open("src/artifacts/training_features.pkl", 'rb') as f:
        training_features = pickle.load(f)
    
    with open("src/artifacts/validation_features.pkl", 'rb') as f:
        validation_features = pickle.load(f)
    
    # Train model
    print("Starting joint training...")
    history = trainer.train(
        training_features=training_features,
        validation_features=validation_features,
        epochs=100,
        batch_size=256
    )
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    
    # Save training history
    with open("src/artifacts/training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print("Joint training completed successfully!")


if __name__ == "__main__":
    main()