#!/usr/bin/env python3
"""
Quick and robust user dataset creation that avoids indexing issues.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import pickle
from src.preprocessing.user_data_preparation import UserDatasetCreator

def create_robust_user_dataset():
    """Create user dataset with proper filtering to avoid indexing issues."""
    
    print("Creating robust user dataset...")
    
    # Load data
    items_df = pd.read_csv("datasets/items.csv")
    users_df = pd.read_csv("datasets/users.csv")
    interactions_df = pd.read_csv("datasets/interactions.csv")
    
    print(f"Original data: {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Get overlapping data to ensure consistency
    users_with_interactions = set(interactions_df['user_id'].unique())
    items_with_interactions = set(interactions_df['product_id'].unique())
    
    # Filter datasets to ensure overlap
    valid_users = users_df[users_df['user_id'].isin(users_with_interactions)]
    valid_items = items_df[items_df['product_id'].isin(items_with_interactions)]
    
    # Sample for faster processing but ensure we have enough data
    sample_users = valid_users.sample(n=min(500, len(valid_users)))
    sample_items = valid_items.sample(n=min(1000, len(valid_items)))
    
    # Filter interactions to only include sampled users and items
    sample_interactions = interactions_df[
        (interactions_df['user_id'].isin(sample_users['user_id'])) &
        (interactions_df['product_id'].isin(sample_items['product_id']))
    ]
    
    print(f"Sampled data: {len(sample_items)} items, {len(sample_users)} users, {len(sample_interactions)} interactions")
    
    # Create dummy item embeddings (replace with actual embeddings if available)
    item_embeddings = {}
    for item_id in sample_items['product_id']:
        item_embeddings[item_id] = np.random.rand(64).astype('float32')
    
    # Initialize dataset creator
    dataset_creator = UserDatasetCreator(max_history_length=20)  # Smaller for faster processing
    
    # Create temporal split
    train_interactions, val_interactions = dataset_creator.create_temporal_split(sample_interactions)
    
    # Create training dataset
    print("Creating training dataset...")
    training_features = dataset_creator.create_training_dataset(
        train_interactions, sample_items, sample_users, item_embeddings,
        negative_samples_per_positive=2
    )
    
    if training_features and len(training_features.get('age', [])) > 0:
        print(f"✅ Successfully created training dataset with {len(training_features['age'])} samples")
        
        # Save the dataset
        os.makedirs("src/artifacts", exist_ok=True)
        with open("src/artifacts/training_features.pkl", 'wb') as f:
            pickle.dump(training_features, f)
        
        # Create smaller validation dataset
        val_sample = val_interactions.sample(n=min(100, len(val_interactions))) if len(val_interactions) > 0 else val_interactions
        val_features = dataset_creator.create_training_dataset(
            val_sample, sample_items, sample_users, item_embeddings,
            negative_samples_per_positive=1
        )
        
        if val_features and len(val_features.get('age', [])) > 0:
            with open("src/artifacts/validation_features.pkl", 'wb') as f:
                pickle.dump(val_features, f)
            print(f"✅ Created validation dataset with {len(val_features['age'])} samples")
        
        # Save dataset stats
        stats = {
            'num_samples': len(training_features['age']),
            'num_positive': int(np.sum(training_features['rating'] > 0.5)),
            'num_negative': int(np.sum(training_features['rating'] <= 0.5)),
            'history_length': training_features['item_history_embeddings'].shape[1],
            'embedding_dim': training_features['item_history_embeddings'].shape[2]
        }
        
        with open("src/artifacts/dataset_stats.txt", 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print("✅ User dataset creation completed successfully!")
        print(f"   Training samples: {stats['num_samples']}")
        print(f"   Positive samples: {stats['num_positive']}")
        print(f"   Negative samples: {stats['num_negative']}")
        
        return True
    else:
        print("❌ Failed to create training features")
        return False

if __name__ == "__main__":
    success = create_robust_user_dataset()
    if success:
        print("\n🎉 Ready for joint training!")
    else:
        print("\n💥 Need to debug further.")