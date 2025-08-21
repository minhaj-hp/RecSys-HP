#!/usr/bin/env python3
"""
Quick setup script to create minimal artifacts for demonstration.
This bypasses the full training pipeline and creates dummy artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.preprocessing.data_loader import DataProcessor

def create_quick_setup():
    """Create minimal artifacts for demo purposes."""
    
    print("=" * 60)
    print("QUICK SETUP - Creating Demo Artifacts")
    print("=" * 60)
    
    # Create artifacts directory
    os.makedirs("src/artifacts", exist_ok=True)
    
    # Load data
    print("Loading data...")
    processor = DataProcessor()
    items_df, users_df, interactions_df = processor.load_data()
    
    print(f"Loaded: {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Build vocabularies
    print("Building vocabularies...")
    processor.build_vocabularies(items_df, users_df, interactions_df)
    processor.save_vocabularies()
    
    # Create dummy item embeddings
    print("Creating item embeddings...")
    item_embeddings = {}
    for item_id in items_df['product_id'].values[:1000]:  # Just first 1000 items
        item_embeddings[item_id] = np.random.rand(64).astype('float32')
    
    np.save("src/artifacts/item_embeddings.npy", item_embeddings)
    print(f"Created embeddings for {len(item_embeddings)} items")
    
    # Create item tower config
    print("Creating item tower config...")
    config = {
        'embedding_dim': 64,
        'hidden_dims': [128, 64],
        'dropout_rate': 0.2
    }
    
    with open("src/artifacts/item_tower_config.txt", 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Create dummy FAISS artifacts
    print("Creating FAISS artifacts...")
    
    # Simple version without actual FAISS for demo
    import faiss
    
    # Create embeddings array
    embedding_array = np.array(list(item_embeddings.values())).astype('float32')
    item_ids = list(item_embeddings.keys())
    
    # Create FAISS index
    index = faiss.IndexFlatIP(64)  # Inner product for cosine similarity
    faiss.normalize_L2(embedding_array)
    index.add(embedding_array)
    
    # Save FAISS index
    faiss.write_index(index, "src/artifacts/faiss_item_index.bin")
    
    # Save metadata
    metadata = {
        'item_id_to_idx': {item_id: idx for idx, item_id in enumerate(item_ids)},
        'idx_to_item_id': {idx: item_id for idx, item_id in enumerate(item_ids)},
        'embedding_dim': 64
    }
    
    with open("src/artifacts/faiss_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    np.save("src/artifacts/faiss_item_embeddings.npy", embedding_array)
    
    print("FAISS artifacts created")
    
    # Create minimal training features for joint training
    print("Creating training features...")
    
    # Sample small subset for demo
    sample_size = 1000
    sample_interactions = interactions_df.sample(n=min(sample_size, len(interactions_df)))
    
    # Create simple training features
    training_features = {
        'age': np.random.normal(35, 10, sample_size).astype('float32'),
        'gender': np.random.choice([0, 1], sample_size).astype('int32'),
        'income': np.random.normal(60000, 20000, sample_size).astype('float32'),
        'item_history_embeddings': np.random.rand(sample_size, 10, 64).astype('float32'),
        'product_id': np.random.choice(list(item_embeddings.keys()), sample_size),
        'category_id': np.random.randint(0, len(processor.category_vocab), sample_size),
        'brand_id': np.random.randint(0, len(processor.brand_vocab), sample_size),
        'price': np.random.uniform(10, 1000, sample_size).astype('float32'),
        'rating': np.random.choice([0.0, 1.0], sample_size).astype('float32')
    }
    
    # Convert product_id to vocab indices
    training_features['product_id'] = np.array([
        processor.item_vocab.get(item, 0) for item in training_features['product_id']
    ])
    
    with open("src/artifacts/training_features.pkl", 'wb') as f:
        pickle.dump(training_features, f)
    
    # Create validation features (smaller)
    val_size = 200
    validation_features = {
        'age': np.random.normal(35, 10, val_size).astype('float32'),
        'gender': np.random.choice([0, 1], val_size).astype('int32'),
        'income': np.random.normal(60000, 20000, val_size).astype('float32'),
        'item_history_embeddings': np.random.rand(val_size, 10, 64).astype('float32'),
        'product_id': np.random.randint(0, len(processor.item_vocab), val_size),
        'category_id': np.random.randint(0, len(processor.category_vocab), val_size),
        'brand_id': np.random.randint(0, len(processor.brand_vocab), val_size),
        'price': np.random.uniform(10, 1000, val_size).astype('float32'),
        'rating': np.random.choice([0.0, 1.0], val_size).astype('float32')
    }
    
    with open("src/artifacts/validation_features.pkl", 'wb') as f:
        pickle.dump(validation_features, f)
    
    print("Training features created")
    
    # Create dataset stats
    stats = {
        'num_samples': sample_size,
        'num_positive': int(np.sum(training_features['rating'] > 0.5)),
        'num_negative': int(np.sum(training_features['rating'] <= 0.5)),
        'history_length': 10,
        'embedding_dim': 64
    }
    
    with open("src/artifacts/dataset_stats.txt", 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print("=" * 60)
    print("QUICK SETUP COMPLETED!")
    print("=" * 60)
    print("Created artifacts:")
    print("✅ Vocabularies")
    print("✅ Item embeddings")
    print("✅ Item tower config")
    print("✅ FAISS index")
    print("✅ Training features")
    print("✅ Validation features")
    print("\nYou can now:")
    print("1. Test the API: python api/main.py")
    print("2. Run the demo: python demo_simple.py")
    print("3. Test recommendation engine: python -m src.inference.recommendation_engine")

def main():
    """Main function."""
    create_quick_setup()

if __name__ == "__main__":
    main()