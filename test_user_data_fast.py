#!/usr/bin/env python3
"""
Fast test for user data preparation with reduced dataset size.
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('src')

from src.preprocessing.user_data_preparation import UserDatasetCreator

def test_user_data_creation():
    """Test user data creation with reduced dataset."""
    
    print("Testing user data creation (fast version)...")
    
    try:
        # Load data
        items_df = pd.read_csv("datasets/items.csv")
        users_df = pd.read_csv("datasets/users.csv")
        interactions_df = pd.read_csv("datasets/interactions.csv")
        
        print(f"Original data sizes:")
        print(f"  Items: {len(items_df)}")
        print(f"  Users: {len(users_df)}")
        print(f"  Interactions: {len(interactions_df)}")
        
        # Sample data to make it faster
        sample_users = users_df.sample(n=min(100, len(users_df)))
        user_ids = set(sample_users['user_id'])
        
        # Filter interactions to only include sample users
        filtered_interactions = interactions_df[interactions_df['user_id'].isin(user_ids)]
        sample_size = min(1000, len(filtered_interactions))
        sample_interactions = filtered_interactions.sample(n=sample_size) if sample_size > 0 else filtered_interactions
        
        # Filter items to only those in interactions
        item_ids = set(sample_interactions['product_id'])
        sample_items = items_df[items_df['product_id'].isin(item_ids)]
        
        print(f"Sampled data sizes:")
        print(f"  Items: {len(sample_items)}")
        print(f"  Users: {len(sample_users)}")
        print(f"  Interactions: {len(sample_interactions)}")
        
        # Initialize dataset creator
        dataset_creator = UserDatasetCreator(max_history_length=10)
        
        # Test temporal split
        print("Testing temporal split...")
        train_interactions, val_interactions = dataset_creator.create_temporal_split(sample_interactions)
        
        print("✅ Temporal split successful!")
        
        # Create dummy item embeddings for testing
        print("Creating dummy item embeddings...")
        item_embeddings = {}
        for item_id in item_ids:
            item_embeddings[item_id] = np.random.rand(64).astype(np.float32)
        
        print(f"Created embeddings for {len(item_embeddings)} items")
        
        # Test user history aggregation
        print("Testing user history aggregation...")
        user_histories = {uid: [list(item_ids)[i % len(item_ids)] for i in range(3)] for uid in list(user_ids)[:10]}
        
        user_aggregated = dataset_creator.aggregate_user_history_embeddings(
            user_histories, item_embeddings, embedding_dim=64
        )
        
        print(f"✅ Created aggregated embeddings for {len(user_aggregated)} users")
        
        # Test user features preparation
        print("Testing user features preparation...")
        user_features = dataset_creator.prepare_user_features(sample_users, user_aggregated)
        
        print(f"✅ User features prepared:")
        for key, arr in user_features.items():
            if hasattr(arr, 'shape'):
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"  {key}: length={len(arr)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_user_data_creation()
    if success:
        print("\n🎉 User data creation test passed!")
    else:
        print("\n💥 User data creation test failed.")