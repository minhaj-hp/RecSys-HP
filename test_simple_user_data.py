#!/usr/bin/env python3
"""
Simple test to identify the core issue with user dataset creation.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from src.preprocessing.data_loader import DataProcessor

def simple_test():
    """Simple test to identify the issue."""
    
    print("=== Simple User Data Test ===")
    
    # Load full data first
    items_df = pd.read_csv("datasets/items.csv")
    users_df = pd.read_csv("datasets/users.csv") 
    interactions_df = pd.read_csv("datasets/interactions.csv")
    
    print(f"Full data: {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Get users who actually have interactions
    users_with_interactions = set(interactions_df['user_id'].unique())
    items_with_interactions = set(interactions_df['product_id'].unique())
    
    print(f"Users with interactions: {len(users_with_interactions)}")
    print(f"Items with interactions: {len(items_with_interactions)}")
    
    # Filter to ensure overlap
    filtered_users = users_df[users_df['user_id'].isin(users_with_interactions)].sample(n=100)
    filtered_items = items_df[items_df['product_id'].isin(items_with_interactions)].sample(n=200)
    filtered_interactions = interactions_df[
        (interactions_df['user_id'].isin(filtered_users['user_id'])) &
        (interactions_df['product_id'].isin(filtered_items['product_id']))
    ]
    
    # Sample only if there are enough interactions
    if len(filtered_interactions) > 500:
        filtered_interactions = filtered_interactions.sample(n=500)
    
    print(f"Filtered data: {len(filtered_items)} items, {len(filtered_users)} users, {len(filtered_interactions)} interactions")
    
    # Test vocabulary building
    processor = DataProcessor()
    processor.build_vocabularies(filtered_items, filtered_users, filtered_interactions)
    
    # Test user history creation
    user_histories = processor.create_user_interaction_history(
        filtered_interactions, filtered_items, max_history_length=10
    )
    
    print(f"User histories created: {len(user_histories)}")
    
    # Create dummy embeddings
    item_embeddings = {}
    for item_id in filtered_items['product_id']:
        item_embeddings[item_id] = np.random.rand(64).astype('float32')
    
    print(f"Item embeddings created: {len(item_embeddings)}")
    
    # Test aggregation
    from src.preprocessing.user_data_preparation import UserDatasetCreator
    creator = UserDatasetCreator()
    
    user_aggregated = creator.aggregate_user_history_embeddings(
        user_histories, item_embeddings, embedding_dim=64
    )
    
    print(f"User aggregated embeddings: {len(user_aggregated)}")
    
    # Test user features preparation
    user_features = creator.prepare_user_features(filtered_users, user_aggregated)
    
    print(f"User features prepared:")
    for key, value in user_features.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} mappings")
        else:
            print(f"  {key}: {len(value)} items")
    
    return True

if __name__ == "__main__":
    simple_test()