#!/usr/bin/env python3
"""
Test the fixed user dataset creation with small sample.
"""

import sys
import os
sys.path.append('src')

from src.preprocessing.user_data_preparation import UserDatasetCreator
import pandas as pd
import numpy as np

def test_fixed_user_dataset():
    """Test the fixed user dataset creation."""
    
    print("Testing fixed user dataset creation...")
    
    try:
        # Initialize dataset creator
        dataset_creator = UserDatasetCreator(max_history_length=10)
        
        # Load data (small sample)
        items_df = pd.read_csv("datasets/items.csv").sample(n=100)
        users_df = pd.read_csv("datasets/users.csv").sample(n=50)
        interactions_df = pd.read_csv("datasets/interactions.csv").sample(n=200)
        
        print(f"Sample sizes: {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
        
        # Load dummy item embeddings
        item_embeddings = {}
        for item_id in items_df['product_id'].values:
            item_embeddings[item_id] = np.random.rand(64).astype('float32')
        
        print(f"Created {len(item_embeddings)} dummy embeddings")
        
        # Create temporal split
        train_interactions, val_interactions = dataset_creator.create_temporal_split(interactions_df)
        
        print(f"Split: {len(train_interactions)} train, {len(val_interactions)} val")
        
        # Test the fixed create_training_dataset function
        print("Testing create_training_dataset...")
        
        training_features = dataset_creator.create_training_dataset(
            train_interactions.sample(n=min(50, len(train_interactions))), 
            items_df, users_df, item_embeddings,
            negative_samples_per_positive=1
        )
        
        if training_features:
            print("✅ Successfully created training features!")
            print(f"   Features created: {list(training_features.keys())}")
            for key, value in training_features.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape {value.shape}")
                else:
                    print(f"   {key}: length {len(value)}")
        else:
            print("❌ No training features created")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_user_dataset()
    if success:
        print("\n🎉 User dataset fix appears to be working!")
    else:
        print("\n💥 Fix needs more work.")