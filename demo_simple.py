#!/usr/bin/env python3
"""
Simple demo of the recommendation system without full training.
Uses basic collaborative filtering with item embeddings.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import pickle
import os

def load_data():
    """Load the datasets."""
    items_df = pd.read_csv("datasets/items.csv")
    users_df = pd.read_csv("datasets/users.csv")
    interactions_df = pd.read_csv("datasets/interactions.csv")
    return items_df, users_df, interactions_df

def create_simple_embeddings(items_df, interactions_df):
    """Create simple item embeddings using SVD on user-item interaction matrix."""
    
    print("Creating simple item embeddings...")
    
    # Create user-item matrix
    user_item_counts = interactions_df.groupby(['user_id', 'product_id']).size().reset_index(name='count')
    
    # Get unique users and items
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_items = sorted(interactions_df['product_id'].unique())
    
    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create matrix
    matrix = np.zeros((len(unique_users), len(unique_items)))
    
    for _, row in user_item_counts.iterrows():
        user_idx = user_to_idx.get(row['user_id'])
        item_idx = item_to_idx.get(row['product_id'])
        if user_idx is not None and item_idx is not None:
            matrix[user_idx, item_idx] = row['count']
    
    # Apply SVD for dimensionality reduction
    print(f"Original matrix shape: {matrix.shape}")
    svd = TruncatedSVD(n_components=64, random_state=42)
    item_embeddings = svd.fit_transform(matrix.T)  # Transpose to get item embeddings
    
    print(f"Item embeddings shape: {item_embeddings.shape}")
    
    # Create item embedding dictionary
    item_embedding_dict = {}
    for item, idx in item_to_idx.items():
        item_embedding_dict[item] = item_embeddings[idx]
    
    return item_embedding_dict, user_to_idx, item_to_idx

class SimpleRecommendationEngine:
    """Simple recommendation engine for demo purposes."""
    
    def __init__(self, items_df, users_df, interactions_df):
        self.items_df = items_df
        self.users_df = users_df
        self.interactions_df = interactions_df
        
        # Create embeddings and mappings
        self.item_embeddings, self.user_to_idx, self.item_to_idx = create_simple_embeddings(
            items_df, interactions_df
        )
        
        # Create user interaction histories
        self.user_histories = defaultdict(list)
        for _, row in interactions_df.iterrows():
            self.user_histories[row['user_id']].append(row['product_id'])
    
    def get_item_info(self, item_id):
        """Get item information."""
        item_row = self.items_df[self.items_df['product_id'] == item_id]
        if len(item_row) > 0:
            item_row = item_row.iloc[0]
            return {
                'product_id': int(item_id),
                'category_code': str(item_row['category_code']),
                'brand': str(item_row['brand']) if pd.notna(item_row['brand']) else 'Unknown',
                'price': float(item_row['price'])
            }
        return None
    
    def get_similar_items(self, item_id, k=10):
        """Get similar items based on embeddings."""
        if item_id not in self.item_embeddings:
            return []
        
        query_embedding = self.item_embeddings[item_id].reshape(1, -1)
        
        # Compute similarities with all items
        similarities = []
        for other_item_id, embedding in self.item_embeddings.items():
            if other_item_id != item_id:
                sim = cosine_similarity(query_embedding, embedding.reshape(1, -1))[0][0]
                similarities.append((other_item_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k with item info
        recommendations = []
        for item_id, score in similarities[:k]:
            item_info = self.get_item_info(item_id)
            if item_info:
                recommendations.append((item_id, score, item_info))
        
        return recommendations
    
    def get_user_recommendations(self, user_id, k=10):
        """Get recommendations for a user based on their history."""
        if user_id not in self.user_histories:
            return []
        
        user_history = self.user_histories[user_id]
        if not user_history:
            return []
        
        # Average embeddings of user's interacted items
        user_embeddings = []
        for item_id in user_history[-10:]:  # Use last 10 interactions
            if item_id in self.item_embeddings:
                user_embeddings.append(self.item_embeddings[item_id])
        
        if not user_embeddings:
            return []
        
        user_profile = np.mean(user_embeddings, axis=0).reshape(1, -1)
        
        # Find similar items
        similarities = []
        user_item_set = set(user_history)
        
        for item_id, embedding in self.item_embeddings.items():
            if item_id not in user_item_set:  # Don't recommend items user already interacted with
                sim = cosine_similarity(user_profile, embedding.reshape(1, -1))[0][0]
                similarities.append((item_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k with item info
        recommendations = []
        for item_id, score in similarities[:k]:
            item_info = self.get_item_info(item_id)
            if item_info:
                recommendations.append((item_id, score, item_info))
        
        return recommendations
    
    def get_demographic_recommendations(self, age, gender, income, k=10):
        """Get recommendations based on user demographics."""
        
        # Find similar users based on demographics
        similar_users = []
        for _, user_row in self.users_df.iterrows():
            age_diff = abs(user_row['age'] - age) / 100.0  # Normalize age difference
            income_diff = abs(user_row['income'] - income) / 100000.0  # Normalize income difference
            gender_match = 1.0 if user_row['gender'] == gender else 0.0
            
            # Simple similarity score
            similarity = (1.0 - age_diff) * 0.3 + (1.0 - income_diff) * 0.3 + gender_match * 0.4
            similar_users.append((user_row['user_id'], similarity))
        
        # Sort by similarity and take top users
        similar_users.sort(key=lambda x: x[1], reverse=True)
        top_users = [user_id for user_id, _ in similar_users[:20]]
        
        # Aggregate items from similar users
        item_scores = defaultdict(float)
        for user_id in top_users:
            if user_id in self.user_histories:
                for item_id in self.user_histories[user_id]:
                    if item_id in self.item_embeddings:
                        item_scores[item_id] += 1.0
        
        # Sort items by score
        top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k with item info
        recommendations = []
        for item_id, score in top_items[:k]:
            item_info = self.get_item_info(item_id)
            if item_info:
                recommendations.append((item_id, score, item_info))
        
        return recommendations

def main():
    """Run the simple demo."""
    
    print("=" * 60)
    print("SIMPLE RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    items_df, users_df, interactions_df = load_data()
    
    print(f"✅ Loaded:")
    print(f"   - {len(items_df)} items")
    print(f"   - {len(users_df)} users")
    print(f"   - {len(interactions_df)} interactions")
    
    # Initialize recommendation engine
    print("\nInitializing recommendation engine...")
    engine = SimpleRecommendationEngine(items_df, users_df, interactions_df)
    print(f"✅ Created embeddings for {len(engine.item_embeddings)} items")
    
    # Demo 1: Item-to-item recommendations
    print("\n" + "=" * 60)
    print("DEMO 1: ITEM-TO-ITEM RECOMMENDATIONS")
    print("=" * 60)
    
    sample_item = list(engine.item_embeddings.keys())[0]
    item_recs = engine.get_similar_items(sample_item, k=5)
    
    print(f"\nItems similar to {sample_item}:")
    sample_info = engine.get_item_info(sample_item)
    if sample_info:
        print(f"Query Item: {sample_info['brand']} - ${sample_info['price']:.2f}")
    
    print(f"\nSimilar Items:")
    for i, (item_id, score, info) in enumerate(item_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.3f})")
    
    # Demo 2: User-based recommendations
    print("\n" + "=" * 60)
    print("DEMO 2: USER-BASED RECOMMENDATIONS")
    print("=" * 60)
    
    sample_user = list(engine.user_histories.keys())[0]
    user_recs = engine.get_user_recommendations(sample_user, k=5)
    
    print(f"\nRecommendations for user {sample_user}:")
    history = engine.user_histories[sample_user][:5]
    print(f"User's recent interactions: {history}")
    
    print(f"\nRecommended Items:")
    for i, (item_id, score, info) in enumerate(user_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.3f})")
    
    # Demo 3: Demographic-based recommendations
    print("\n" + "=" * 60)
    print("DEMO 3: DEMOGRAPHIC-BASED RECOMMENDATIONS")
    print("=" * 60)
    
    # Sample demographics
    demo_age = 30
    demo_gender = "male"
    demo_income = 60000
    
    demo_recs = engine.get_demographic_recommendations(demo_age, demo_gender, demo_income, k=5)
    
    print(f"\nRecommendations for demographics:")
    print(f"Age: {demo_age}, Gender: {demo_gender}, Income: ${demo_income:,}")
    
    print(f"\nRecommended Items:")
    for i, (item_id, score, info) in enumerate(demo_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.1f})")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("\nThis simple demo shows basic recommendation functionality.")
    print("For the full two-tower architecture, run: python run_training_pipeline.py")

if __name__ == "__main__":
    main()