#!/usr/bin/env python3
"""
Setup script for demographic clustering.

This script creates demographic clusters from existing user data
to improve cold-start recommendations.
"""

import sys
import os

# Add src to path
sys.path.append('src')
from src.inference.demographic_clustering import DemographicClusterer

def main():
    """Set up demographic clustering."""
    print("=== Setting up Demographic Clustering for Cold-Start Recommendations ===")
    
    # Check if data files exist
    required_files = [
        "datasets/users.csv",
        "datasets/interactions.csv", 
        "datasets/items.csv"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: Required file {file_path} not found")
            return
    
    print("✅ All required data files found")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("src/artifacts", exist_ok=True)
    
    try:
        # Initialize demographic clusterer
        clusterer = DemographicClusterer(artifacts_path="src/artifacts/")
        
        # Fit clusters on user data (12 clusters seems optimal)
        print("\nFitting demographic clusters on user data...")
        cluster_labels = clusterer.fit_clusters(
            n_clusters=12,
            users_csv_path="datasets/users.csv",
            interactions_csv_path="datasets/interactions.csv", 
            items_csv_path="datasets/items.csv"
        )
        
        # Print cluster summary
        clusterer.print_cluster_summary()
        
        # Test clustering with sample users
        print("\n=== Testing Cluster Predictions ===")
        
        test_users = [
            {'age': 22, 'gender': 'male', 'income': 35000, 'profession': 'Technology', 'location': 'Urban'},
            {'age': 28, 'gender': 'female', 'income': 65000, 'profession': 'Healthcare', 'location': 'Suburban'},
            {'age': 35, 'gender': 'male', 'income': 95000, 'profession': 'Finance', 'location': 'Urban'},
            {'age': 45, 'gender': 'female', 'income': 55000, 'profession': 'Education', 'location': 'Rural'},
            {'age': 55, 'gender': 'male', 'income': 120000, 'profession': 'Technology', 'location': 'Urban'}
        ]
        
        for i, user in enumerate(test_users):
            cluster_id = clusterer.predict_cluster(**user)
            categories = clusterer.get_category_recommendations_for_user(**user, top_k=3)
            
            print(f"\nTest User {i+1}: {user['age']}y {user['gender']} ${user['income']} {user['profession']}")
            print(f"  → Cluster {cluster_id}")
            print(f"  → Top categories: {[(cat, f'{prob:.2f}') for cat, prob in categories]}")
        
        print(f"\n✅ Demographic clustering setup completed successfully!")
        print(f"   Cluster data saved to src/artifacts/demographic_clusters.pkl")
        print(f"   The recommendation engine will now provide better cold-start recommendations")
        
    except Exception as e:
        print(f"❌ Error setting up demographic clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()