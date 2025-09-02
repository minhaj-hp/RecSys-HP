#!/usr/bin/env python3
"""
Two-Phase Training Pipeline Runner

This script orchestrates the complete 2-phase training approach:
1. Phase 1: Item tower pretraining on item features
2. Phase 2: Joint training of user tower + fine-tuning pre-trained item tower

Usage:
    python run_2phase_training.py
"""

import os
import sys
import time
import pickle
import numpy as np
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.item_pretraining import ItemTowerPretrainer
from src.training.joint_training import JointTrainer
from src.preprocessing.data_loader import DataProcessor
from src.inference.faiss_index import FAISSItemIndex


def run_phase1_item_pretraining():
    """Phase 1: Pre-train the item tower."""
    
    print("\n" + "="*60)
    print("PHASE 1: ITEM TOWER PRETRAINING")
    print("="*60)
    
    # Initialize components
    data_processor = DataProcessor()
    pretrainer = ItemTowerPretrainer(
        embedding_dim=128,
        hidden_dims=[256, 128],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Prepare data
    print("Preparing item data...")
    dataset, data_processor, price_normalizer = pretrainer.prepare_data(data_processor)
    
    # Build model
    print("Building item tower...")
    model = pretrainer.build_model(
        item_vocab_size=len(data_processor.item_vocab),
        category_vocab_size=len(data_processor.category_vocab),
        brand_vocab_size=len(data_processor.brand_vocab),
        price_normalizer=price_normalizer
    )
    
    # Train model
    print("Training item tower (Phase 1)...")
    start_time = time.time()
    history = pretrainer.train(dataset, epochs=50)
    phase1_time = time.time() - start_time
    
    # Generate embeddings
    print("Generating item embeddings...")
    item_embeddings = pretrainer.generate_item_embeddings(dataset, data_processor)
    
    # Save artifacts
    print("Saving Phase 1 artifacts...")
    os.makedirs("src/artifacts", exist_ok=True)
    data_processor.save_vocabularies()
    pretrainer.save_model()
    
    # Save embeddings for FAISS index
    np.save("src/artifacts/item_embeddings.npy", item_embeddings)
    
    # Build FAISS index
    print("Building FAISS index...")
    faiss_index = FAISSItemIndex()
    faiss_index.build_index(item_embeddings)
    faiss_index.save_index("src/artifacts/")
    
    print(f"✅ Phase 1 completed in {phase1_time:.2f} seconds")
    print(f"   - Items processed: {len(item_embeddings)}")
    print(f"   - Final loss: {history.history['total_loss'][-1]:.4f}")
    
    return data_processor


def run_phase2_joint_training(data_processor: DataProcessor):
    """Phase 2: Joint training with pre-trained item tower."""
    
    print("\n" + "="*60)
    print("PHASE 2: JOINT TRAINING")
    print("="*60)
    
    # Initialize joint trainer
    trainer = JointTrainer(
        embedding_dim=128,
        user_learning_rate=0.001,
        item_learning_rate=0.0001,  # Lower LR for pre-trained item tower
        rating_weight=1.0,
        retrieval_weight=0.5
    )
    
    # Load pre-trained item tower
    print("Loading pre-trained item tower...")
    trainer.load_pre_trained_item_tower()
    
    # Build user tower
    print("Building user tower...")
    trainer.build_user_tower(max_history_length=50)
    
    # Build complete two-tower model
    print("Building complete two-tower model...")
    trainer.build_two_tower_model()
    
    # Prepare training data
    print("Preparing user interaction data...")
    
    # Check if training features already exist
    if os.path.exists("src/artifacts/training_features.pkl"):
        print("Loading existing training features...")
        with open("src/artifacts/training_features.pkl", 'rb') as f:
            training_features = pickle.load(f)
        with open("src/artifacts/validation_features.pkl", 'rb') as f:
            validation_features = pickle.load(f)
    else:
        # Generate training features
        print("Generating training features...")
        training_features, validation_features = data_processor.prepare_training_data()
        
        # Save features
        with open("src/artifacts/training_features.pkl", 'wb') as f:
            pickle.dump(training_features, f)
        with open("src/artifacts/validation_features.pkl", 'wb') as f:
            pickle.dump(validation_features, f)
    
    # Train joint model
    print("Starting joint training (Phase 2)...")
    start_time = time.time()
    history = trainer.train(
        training_features=training_features,
        validation_features=validation_features,
        epochs=100,
        batch_size=256
    )
    phase2_time = time.time() - start_time
    
    # Save final model
    print("Saving final two-tower model...")
    trainer.save_model()
    
    # Save training history
    with open("src/artifacts/joint_training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    print(f"✅ Phase 2 completed in {phase2_time:.2f} seconds")
    print(f"   - Best validation loss: {min(history['val_total_loss']):.4f}")
    print(f"   - Epochs trained: {len(history['total_loss'])}")
    
    return history


def main():
    """Main function to run complete 2-phase training pipeline."""
    
    print("🚀 STARTING 2-PHASE TRAINING PIPELINE")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    total_start_time = time.time()
    
    try:
        # Phase 1: Item tower pretraining
        data_processor = run_phase1_item_pretraining()
        
        # Phase 2: Joint training
        history = run_phase2_joint_training(data_processor)
        
        # Final summary
        total_time = time.time() - total_start_time
        
        print("\n" + "="*60)
        print("🎉 2-PHASE TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Artifacts saved in: src/artifacts/")
        print("\nKey files generated:")
        print("  - item_tower_weights: Pre-trained item embeddings")
        print("  - item_tower_weights_finetuned_best: Fine-tuned item tower")
        print("  - user_tower_weights_best: Trained user tower") 
        print("  - rating_model_weights_best: Rating prediction model")
        print("  - faiss_index.index: Item similarity index")
        print("  - vocabularies.pkl: Feature vocabularies")
        
        print(f"\n🔥 Final validation loss: {min(history['val_total_loss']):.4f}")
        print("\n✅ Ready to run inference with api/main.py!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()