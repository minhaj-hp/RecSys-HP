"""
Demographic clustering system for cold-start recommendations.

This module creates demographic clusters from existing users and provides
category preference mappings for new users with no interaction history.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import pickle
import os


class DemographicClusterer:
    """Clusters users by demographics and learns category preferences."""
    
    def __init__(self, artifacts_path: str = "src/artifacts/"):
        self.artifacts_path = artifacts_path
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_preferences = {}
        self.cluster_centers_original = None
        self.demographic_mappings = self._load_demographic_mappings()
        
        # Try to load pre-computed clusters
        if os.path.exists(f"{artifacts_path}/demographic_clusters.pkl"):
            self.load_clusters()
        else:
            print("Demographic clusters not found. Need to fit on user data first.")
    
    def _load_demographic_mappings(self) -> Dict:
        """Load demographic category mappings."""
        return {
            'profession_map': {
                "Technology": 0, "Healthcare": 1, "Education": 2, "Finance": 3,
                "Retail": 4, "Manufacturing": 5, "Services": 6, "Other": 7
            },
            'location_map': {"Urban": 0, "Suburban": 1, "Rural": 2},
            'education_map': {
                "High School": 0, "Some College": 1, "Bachelor's": 2, 
                "Master's": 3, "PhD+": 4
            },
            'marital_map': {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
        }
    
    def _categorize_age(self, age: float) -> int:
        """Categorize age into demographic groups."""
        if age < 18: return 0  # Teen
        elif age < 26: return 1  # Young Adult  
        elif age < 36: return 2  # Adult
        elif age < 51: return 3  # Middle Age
        elif age < 66: return 4  # Mature
        else: return 5  # Senior
    
    def _categorize_income(self, income: float) -> int:
        """Categorize income into quintiles."""
        if income < 30000: return 0
        elif income < 50000: return 1
        elif income < 75000: return 2
        elif income < 100000: return 3
        else: return 4
    
    def prepare_demographic_features(self, user_data: pd.DataFrame) -> np.ndarray:
        """Prepare demographic features for clustering."""
        features = []
        
        for _, user in user_data.iterrows():
            # Use both categorical and continuous features
            feature_vector = [
                user['age'],  # Continuous age
                self._categorize_age(user['age']),  # Categorical age
                1 if user['gender'].lower() == 'male' else 0,  # Gender
                user['income'],  # Continuous income
                self._categorize_income(user['income']),  # Categorical income
                self.demographic_mappings['profession_map'].get(user.get('profession', 'Other'), 7),
                self.demographic_mappings['location_map'].get(user.get('location', 'Urban'), 0),
                self.demographic_mappings['education_map'].get(user.get('education_level', 'High School'), 0),
                self.demographic_mappings['marital_map'].get(user.get('marital_status', 'Single'), 0),
                # Interaction terms
                user['age'] * user['income'] / 1000000,  # Age-income interaction
                user['age'] * (1 if user['gender'].lower() == 'male' else 0),  # Age-gender interaction
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit_clusters(self, n_clusters: int = 12, users_csv_path: str = "datasets/users.csv", 
                    interactions_csv_path: str = "datasets/interactions.csv",
                    items_csv_path: str = "datasets/items.csv"):
        """Fit demographic clusters on existing user data and learn preferences."""
        
        print(f"Loading user data for demographic clustering...")
        
        # Load datasets
        users_df = pd.read_csv(users_csv_path)
        interactions_df = pd.read_csv(interactions_csv_path) 
        items_df = pd.read_csv(items_csv_path)
        
        print(f"Loaded {len(users_df)} users, {len(interactions_df)} interactions")
        
        # Filter users with sufficient interaction history
        user_interaction_counts = interactions_df['user_id'].value_counts()
        users_with_history = user_interaction_counts[user_interaction_counts >= 5].index
        filtered_users = users_df[users_df['user_id'].isin(users_with_history)]
        
        print(f"Using {len(filtered_users)} users with ≥5 interactions for clustering")
        
        # Prepare demographic features
        demographic_features = self.prepare_demographic_features(filtered_users)
        
        # Fit scaler and transform features
        demographic_features_scaled = self.scaler.fit_transform(demographic_features)
        
        # Fit KMeans clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(demographic_features_scaled)
        
        # Store original centers for interpretation
        self.cluster_centers_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        print(f"Created {n_clusters} demographic clusters")
        
        # Learn category preferences for each cluster
        self._learn_cluster_preferences(filtered_users, cluster_labels, interactions_df, items_df)
        
        # Save clusters
        self.save_clusters()
        
        return cluster_labels
    
    def _learn_cluster_preferences(self, users_df: pd.DataFrame, cluster_labels: np.ndarray,
                                 interactions_df: pd.DataFrame, items_df: pd.DataFrame):
        """Learn category preferences for each demographic cluster."""
        
        print("Learning category preferences for each cluster...")
        
        # Group users by cluster
        users_df = users_df.copy()
        users_df['cluster'] = cluster_labels
        
        self.cluster_preferences = {}
        
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_users = users_df[users_df['cluster'] == cluster_id]['user_id'].values
            
            # Get all interactions for users in this cluster
            cluster_interactions = interactions_df[interactions_df['user_id'].isin(cluster_users)]
            
            # Get item categories for these interactions
            cluster_items = cluster_interactions.merge(items_df, on='product_id', how='left')
            
            # Count category preferences
            category_counts = Counter()
            total_interactions = len(cluster_items)
            
            for _, item in cluster_items.iterrows():
                category = item['category_code']
                if pd.notna(category):
                    # Use 2-level category hierarchy
                    if '.' in str(category):
                        parts = str(category).split('.')
                        if len(parts) >= 2:
                            category = f"{parts[0]}.{parts[1]}"
                        else:
                            category = parts[0]
                    category_counts[category] += 1
            
            # Convert to preferences (normalized probabilities)
            cluster_preferences = {}
            for category, count in category_counts.items():
                cluster_preferences[category] = count / total_interactions
            
            # Store cluster info
            cluster_center = self.cluster_centers_original[cluster_id]
            self.cluster_preferences[cluster_id] = {
                'category_preferences': cluster_preferences,
                'user_count': len(cluster_users),
                'total_interactions': total_interactions,
                'cluster_center': {
                    'age': cluster_center[0],
                    'age_category': int(cluster_center[1]),
                    'gender': 'male' if cluster_center[2] > 0.5 else 'female',
                    'income': cluster_center[3],
                    'income_category': int(cluster_center[4]),
                    'profession': cluster_center[5],
                    'location': cluster_center[6],
                    'education': cluster_center[7],
                    'marital_status': cluster_center[8]
                },
                'top_categories': sorted(cluster_preferences.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
            }
            
            print(f"Cluster {cluster_id}: {len(cluster_users)} users, "
                  f"top categories: {[cat for cat, _ in self.cluster_preferences[cluster_id]['top_categories'][:3]]}")
    
    def predict_cluster(self, age: int, gender: str, income: float,
                       profession: str = "Other", location: str = "Urban",
                       education_level: str = "High School", marital_status: str = "Single") -> int:
        """Predict demographic cluster for a new user."""
        
        if self.kmeans is None or self.scaler is None:
            raise ValueError("Clusters not fitted. Call fit_clusters() first.")
        
        # Prepare user features
        user_features = np.array([[
            age,  # Continuous age
            self._categorize_age(age),  # Categorical age  
            1 if gender.lower() == 'male' else 0,  # Gender
            income,  # Continuous income
            self._categorize_income(income),  # Categorical income
            self.demographic_mappings['profession_map'].get(profession, 7),
            self.demographic_mappings['location_map'].get(location, 0), 
            self.demographic_mappings['education_map'].get(education_level, 0),
            self.demographic_mappings['marital_map'].get(marital_status, 0),
            # Interaction terms
            age * income / 1000000,  # Age-income interaction
            age * (1 if gender.lower() == 'male' else 0),  # Age-gender interaction
        ]])
        
        # Scale features and predict cluster
        user_features_scaled = self.scaler.transform(user_features)
        cluster_id = self.kmeans.predict(user_features_scaled)[0]
        
        return int(cluster_id)
    
    def get_cluster_category_preferences(self, cluster_id: int) -> Dict:
        """Get category preferences for a specific cluster."""
        return self.cluster_preferences.get(cluster_id, {})
    
    def get_category_recommendations_for_user(self, age: int, gender: str, income: float,
                                            profession: str = "Other", location: str = "Urban",
                                            education_level: str = "High School", 
                                            marital_status: str = "Single", 
                                            top_k: int = 5) -> List[Tuple[str, float]]:
        """Get category recommendations for a new user based on their predicted cluster."""
        
        # Predict user's cluster
        cluster_id = self.predict_cluster(age, gender, income, profession, location, 
                                        education_level, marital_status)
        
        # Get cluster preferences
        cluster_info = self.get_cluster_category_preferences(cluster_id)
        
        if not cluster_info or 'category_preferences' not in cluster_info:
            return []
        
        # Return top categories with their preference scores
        category_prefs = cluster_info['category_preferences']
        sorted_categories = sorted(category_prefs.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_categories[:top_k]
    
    def save_clusters(self):
        """Save fitted clusters to disk."""
        cluster_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler, 
            'cluster_preferences': self.cluster_preferences,
            'cluster_centers_original': self.cluster_centers_original,
            'demographic_mappings': self.demographic_mappings
        }
        
        with open(f"{self.artifacts_path}/demographic_clusters.pkl", 'wb') as f:
            pickle.dump(cluster_data, f)
        
        print(f"Demographic clusters saved to {self.artifacts_path}/demographic_clusters.pkl")
    
    def load_clusters(self):
        """Load pre-fitted clusters from disk."""
        try:
            with open(f"{self.artifacts_path}/demographic_clusters.pkl", 'rb') as f:
                cluster_data = pickle.load(f)
            
            self.kmeans = cluster_data['kmeans']
            self.scaler = cluster_data['scaler']
            self.cluster_preferences = cluster_data['cluster_preferences']
            self.cluster_centers_original = cluster_data['cluster_centers_original']
            self.demographic_mappings = cluster_data.get('demographic_mappings', self.demographic_mappings)
            
            print(f"Loaded demographic clusters with {self.kmeans.n_clusters} clusters")
            
        except Exception as e:
            print(f"Error loading demographic clusters: {e}")
            raise
    
    def print_cluster_summary(self):
        """Print a summary of all clusters and their characteristics."""
        
        if not self.cluster_preferences:
            print("No cluster preferences available. Fit clusters first.")
            return
        
        print("\n=== Demographic Cluster Summary ===")
        
        for cluster_id, info in self.cluster_preferences.items():
            center = info['cluster_center']
            print(f"\nCluster {cluster_id}:")
            print(f"  Users: {info['user_count']}")
            print(f"  Demographics: {center['age']:.1f}y {center['gender']} ${center['income']:.0f}")
            print(f"  Top categories: {[f'{cat}({prob:.2f})' for cat, prob in info['top_categories'][:3]]}")


def main():
    """Demo and fit demographic clusters."""
    print("=== Demographic Clustering for Cold-Start Recommendations ===")
    
    # Initialize clusterer
    clusterer = DemographicClusterer()
    
    # Fit clusters on user data
    cluster_labels = clusterer.fit_clusters(n_clusters=12)
    
    # Print cluster summary
    clusterer.print_cluster_summary()
    
    # Test predictions for sample users
    print("\n=== Sample Predictions ===")
    
    test_users = [
        {'age': 25, 'gender': 'male', 'income': 45000, 'profession': 'Technology'},
        {'age': 35, 'gender': 'female', 'income': 85000, 'profession': 'Healthcare'},
        {'age': 45, 'gender': 'male', 'income': 120000, 'profession': 'Finance'},
    ]
    
    for user in test_users:
        cluster_id = clusterer.predict_cluster(**user)
        categories = clusterer.get_category_recommendations_for_user(**user, top_k=3)
        
        print(f"\nUser: {user['age']}y {user['gender']} ${user['income']} {user['profession']}")
        print(f"  Predicted cluster: {cluster_id}")
        print(f"  Recommended categories: {categories}")
    
    print("\n✅ Demographic clustering completed successfully!")


if __name__ == "__main__":
    main()