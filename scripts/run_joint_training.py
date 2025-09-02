#!/usr/bin/env python3
"""
Single Joint Training Pipeline Runner

This script orchestrates the single-phase joint training approach:
- Trains user tower and item tower simultaneously from scratch
- No pre-training phase - direct end-to-end optimization
- Supports both regular and fast training modes

Usage:
    python run_joint_training.py [--fast]
"""

import os
import sys
import time
import pickle
import argparse
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.fast_joint_training import FastJointTrainer
from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower, TwoTowerModel
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset
from src.inference.faiss_index import FAISSItemIndex
import tensorflow as tf
import numpy as np


class SingleJointTrainer:
    """Complete single-phase joint training from scratch."""
    
    def __init__(self):
        self.item_tower = None
        self.user_tower = None
        self.model = None
        self.data_processor = None
        
        # Training hyperparameters
        self.embedding_dim = 128
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 80
        self.patience = 20
    
    def prepare_data(self):
        """Prepare all training data from scratch."""
        
        print("Loading and preparing data...")
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Check if preprocessed data exists
        if os.path.exists("src/artifacts/training_features.pkl"):
            print("Loading existing preprocessed data...")
            
            # Load vocabularies
            self.data_processor.load_vocabularies("src/artifacts/vocabularies.pkl")
            
            # Load training features
            with open("src/artifacts/training_features.pkl", 'rb') as f:
                training_features = pickle.load(f)
            with open("src/artifacts/validation_features.pkl", 'rb') as f:
                validation_features = pickle.load(f)
        else:
            print("Preprocessing data from scratch...")
            
            # Load raw data and build vocabularies
            items_df, users_df, interactions_df = self.data_processor.load_data()
            self.data_processor.build_vocabularies(items_df, users_df, interactions_df)
            
            # Generate training features
            training_features, validation_features = self.data_processor.prepare_training_data()
            
            # Save for future use
            os.makedirs("src/artifacts", exist_ok=True)
            self.data_processor.save_vocabularies()
            
            with open("src/artifacts/training_features.pkl", 'wb') as f:
                pickle.dump(training_features, f)
            with open("src/artifacts/validation_features.pkl", 'wb') as f:
                pickle.dump(validation_features, f)
        
        print(f"Training samples: {len(training_features['rating']):,}")
        print(f"Validation samples: {len(validation_features['rating']):,}")
        
        return training_features, validation_features
    
    def build_models(self):
        """Build both towers from scratch."""
        
        print("Building item tower...")
        self.item_tower = ItemTower(
            item_vocab_size=len(self.data_processor.item_vocab),
            category_vocab_size=len(self.data_processor.category_vocab),
            brand_vocab_size=len(self.data_processor.brand_vocab),
            embedding_dim=self.embedding_dim,
            hidden_dims=[256, 128],
            dropout_rate=0.2
        )
        
        print("Building user tower...")
        self.user_tower = UserTower(
            max_history_length=50,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64],  # Match trained architecture
            dropout_rate=0.2
        )
        
        print("Building complete two-tower model...")
        self.model = TwoTowerModel(
            item_tower=self.item_tower,
            user_tower=self.user_tower,
            rating_weight=1.0,
            retrieval_weight=0.5
        )
        
        print("Models initialized successfully")
    
    def train_joint_model(self, training_features: Dict, validation_features: Dict):
        """Train both towers jointly from scratch."""
        
        print(f"Starting single-phase joint training...")
        print(f"Configuration: {self.epochs} epochs, batch size {self.batch_size}")
        
        # Create datasets
        train_dataset = create_tf_dataset(training_features, self.batch_size)
        val_dataset = create_tf_dataset(validation_features, self.batch_size)
        
        # Setup optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
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
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Training phase
            epoch_losses = {'total_loss': [], 'rating_loss': [], 'retrieval_loss': []}
            
            for batch in train_dataset:
                with tf.GradientTape() as tape:
                    # Forward pass
                    user_embeddings = self.user_tower(batch, training=True)
                    item_embeddings = self.item_tower(batch, training=True)
                    
                    # Rating prediction
                    concatenated = tf.concat([user_embeddings, item_embeddings], axis=-1)
                    rating_predictions = self.model.rating_model(concatenated, training=True)
                    
                    # Rating loss
                    rating_loss = self.model.rating_task(
                        labels=batch["rating"],
                        predictions=rating_predictions
                    )
                    
                    # Retrieval loss (dot product similarity)
                    similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
                    retrieval_loss = self.model.retrieval_loss(
                        batch["rating"], 
                        tf.nn.sigmoid(similarities)
                    )
                    
                    # Combined loss
                    total_loss = (
                        self.model.rating_weight * rating_loss +
                        self.model.retrieval_weight * retrieval_loss
                    )
                
                # Compute and apply gradients
                all_variables = (
                    self.user_tower.trainable_variables +
                    self.item_tower.trainable_variables +
                    self.model.rating_model.trainable_variables
                )
                gradients = tape.gradient(total_loss, all_variables)
                optimizer.apply_gradients(zip(gradients, all_variables))
                
                # Track losses
                epoch_losses['total_loss'].append(total_loss)
                epoch_losses['rating_loss'].append(rating_loss)
                epoch_losses['retrieval_loss'].append(retrieval_loss)
            
            # Validation phase
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
                
                similarities = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)
                retrieval_loss = self.model.retrieval_loss(
                    batch["rating"], 
                    tf.nn.sigmoid(similarities)
                )
                
                total_loss = (
                    self.model.rating_weight * rating_loss +
                    self.model.retrieval_weight * retrieval_loss
                )
                
                val_losses['total_loss'].append(total_loss)
                val_losses['rating_loss'].append(rating_loss)
                val_losses['retrieval_loss'].append(retrieval_loss)
            
            # Calculate average losses
            avg_train_losses = {k: tf.reduce_mean(v).numpy() for k, v in epoch_losses.items()}
            avg_val_losses = {k: tf.reduce_mean(v).numpy() for k, v in val_losses.items()}
            
            # Update history
            for key in history.keys():
                if key.startswith('val_'):
                    history[key].append(avg_val_losses[key.replace('val_', '')])
                else:
                    history[key].append(avg_train_losses[key])
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Time: {epoch_time:.1f}s | Train: {avg_train_losses['total_loss']:.4f} | Val: {avg_val_losses['total_loss']:.4f}")
            print(f"  Rating: {avg_val_losses['rating_loss']:.4f} | Retrieval: {avg_val_losses['retrieval_loss']:.4f}")
            
            # Early stopping and best model saving
            if avg_val_losses['total_loss'] < best_val_loss:
                best_val_loss = avg_val_losses['total_loss']
                patience_counter = 0
                self.save_model("_best")
                print("  ✅ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        print("Joint training completed!")
        return history
    
    def generate_item_embeddings(self, training_features: Dict):
        """Generate item embeddings for FAISS index."""
        
        print("Generating item embeddings...")
        
        # Get all unique items from training data
        unique_items = np.unique(training_features['product_id'])
        item_embeddings = {}
        
        # Process in batches
        batch_size = 1000
        for i in range(0, len(unique_items), batch_size):
            batch_items = unique_items[i:i+batch_size]
            
            # Create batch features
            batch_features = {
                'product_id': batch_items,
                'category_id': training_features['category_id'][:len(batch_items)],
                'brand_id': training_features['brand_id'][:len(batch_items)],
                'price': training_features['price'][:len(batch_items)]
            }
            
            # Convert to tensors
            batch_tensors = {k: tf.constant(v) for k, v in batch_features.items()}
            
            # Get embeddings
            embeddings = self.item_tower(batch_tensors, training=False)
            
            # Store embeddings
            for j, item_id in enumerate(batch_items):
                # Map back from vocab index to actual item ID
                actual_item_id = item_id  # Assuming direct mapping
                item_embeddings[actual_item_id] = embeddings[j].numpy()
        
        print(f"Generated embeddings for {len(item_embeddings)} items")
        return item_embeddings
    
    def save_model(self, suffix=""):
        """Save trained models."""
        
        save_path = "src/artifacts/"
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        self.user_tower.save_weights(f"{save_path}/user_tower_weights{suffix}")
        self.item_tower.save_weights(f"{save_path}/item_tower_weights_finetuned{suffix}")
        self.model.rating_model.save_weights(f"{save_path}/rating_model_weights{suffix}")
        
        # Save item tower config for inference
        with open(f"{save_path}/item_tower_config.txt", 'w') as f:
            f.write(f"embedding_dim: {self.embedding_dim}\n")
            f.write(f"hidden_dims: [256, 128]\n")  # Item tower architecture
            f.write(f"dropout_rate: 0.2\n")
        
        if not suffix:
            print("Final model saved")


def run_fast_joint_training():
    """Run fast optimized joint training."""
    
    print("\n" + "="*60)
    print("FAST JOINT TRAINING MODE")
    print("="*60)
    
    # Initialize fast trainer
    trainer = FastJointTrainer()
    
    # Check if we need to prepare data first
    if not os.path.exists("src/artifacts/training_features.pkl"):
        print("Preparing data first...")
        single_trainer = SingleJointTrainer()
        training_features, validation_features = single_trainer.prepare_data()
    
    # Run fast training
    trainer.load_components()
    
    print("Loading training data...")
    with open("src/artifacts/training_features.pkl", 'rb') as f:
        training_features = pickle.load(f)
    with open("src/artifacts/validation_features.pkl", 'rb') as f:
        validation_features = pickle.load(f)
    
    start_time = time.time()
    trainer.train_fast(training_features, validation_features)
    training_time = time.time() - start_time
    
    # Generate embeddings and build FAISS index
    print("Building FAISS index...")
    # Use single trainer for embedding generation
    single_trainer = SingleJointTrainer()
    single_trainer.data_processor = DataProcessor()
    single_trainer.data_processor.load_vocabularies("src/artifacts/vocabularies.pkl")
    single_trainer.item_tower = trainer.item_tower
    
    item_embeddings = single_trainer.generate_item_embeddings(training_features)
    
    faiss_index = FAISSItemIndex()
    faiss_index.build_index(item_embeddings)
    faiss_index.save_index("src/artifacts/")
    
    return training_time


def run_regular_joint_training():
    """Run regular comprehensive joint training."""
    
    print("\n" + "="*60)
    print("REGULAR JOINT TRAINING MODE")
    print("="*60)
    
    # Initialize trainer
    trainer = SingleJointTrainer()
    
    # Prepare data
    training_features, validation_features = trainer.prepare_data()
    
    # Build models from scratch
    trainer.build_models()
    
    # Train joint model
    start_time = time.time()
    history = trainer.train_joint_model(training_features, validation_features)
    training_time = time.time() - start_time
    
    # Generate item embeddings
    item_embeddings = trainer.generate_item_embeddings(training_features)
    
    # Build FAISS index
    print("Building FAISS index...")
    faiss_index = FAISSItemIndex()
    faiss_index.build_index(item_embeddings)
    faiss_index.save_index("src/artifacts/")
    
    # Save final model
    trainer.save_model()
    
    # Save training history
    with open("src/artifacts/single_joint_training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    return training_time, history


def main():
    """Main function to run single joint training pipeline."""
    
    parser = argparse.ArgumentParser(description='Single Joint Training Pipeline')
    parser.add_argument('--fast', action='store_true', help='Use fast training mode')
    args = parser.parse_args()
    
    print("🚀 STARTING SINGLE JOINT TRAINING PIPELINE")
    print(f"Working directory: {os.getcwd()}")
    print(f"Training mode: {'FAST' if args.fast else 'REGULAR'}")
    
    total_start_time = time.time()
    
    try:
        if args.fast:
            training_time = run_fast_joint_training()
            history = None
        else:
            training_time, history = run_regular_joint_training()
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*60)
        print("🎉 SINGLE JOINT TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Artifacts saved in: src/artifacts/")
        
        print("\nKey files generated:")
        print("  - user_tower_weights_best: Trained user tower")
        print("  - item_tower_weights_finetuned_best: Trained item tower") 
        print("  - rating_model_weights_best: Rating prediction model")
        print("  - faiss_index.index: Item similarity index")
        print("  - vocabularies.pkl: Feature vocabularies")
        
        if history:
            print(f"\n🔥 Best validation loss: {min(history['val_total_loss']):.4f}")
        
        print(f"\n🎯 Training approach: Single-phase joint optimization")
        print("✅ Ready to run inference with api/main.py!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()