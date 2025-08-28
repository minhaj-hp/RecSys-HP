#!/usr/bin/env python3
"""
Real user selector utility - extracts genuine user profiles and interaction histories
from the actual datasets instead of using synthetic data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random
from collections import defaultdict


class RealUserSelector:
    """Extracts real user profiles with genuine interaction histories."""
    
    def __init__(self, 
                 users_path: str = "datasets/users.csv",
                 interactions_path: str = "datasets/interactions.csv",
                 items_path: str = "datasets/items.csv"):
        
        self.users_path = users_path
        self.interactions_path = interactions_path  
        self.items_path = items_path
        
        # Load datasets
        print("Loading real datasets...")
        self.users_df = pd.read_csv(users_path)
        self.interactions_df = pd.read_csv(interactions_path)
        self.items_df = pd.read_csv(items_path) if items_path else None
        
        print(f"Loaded {len(self.users_df)} users, {len(self.interactions_df)} interactions")
        
        # Cache interaction statistics
        self._cache_interaction_stats()
    
    def _cache_interaction_stats(self):
        """Cache interaction statistics per user for faster lookup."""
        print("Caching user interaction statistics...")
        
        self.user_stats = defaultdict(lambda: {
            'total_interactions': 0,
            'views': 0, 
            'cart': 0,
            'purchase': 0,
            'item_ids': [],
            'event_types': []
        })
        
        # Group interactions by user
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['user_id']
            event_type = interaction['event_type']
            item_id = interaction['product_id']
            
            stats = self.user_stats[user_id]
            stats['total_interactions'] += 1
            stats['item_ids'].append(item_id)
            stats['event_types'].append(event_type)
            
            # Count by event type
            if event_type == 'view':
                stats['views'] += 1
            elif event_type in ['cart', 'add_to_cart']:
                stats['cart'] += 1
            elif event_type in ['purchase', 'buy']:
                stats['purchase'] += 1
        
        # Convert lists to unique items for history
        for user_id in self.user_stats:
            stats = self.user_stats[user_id]
            # Keep chronological order but remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item_id in stats['item_ids']:
                if item_id not in seen:
                    seen.add(item_id)
                    unique_items.append(item_id)
            stats['unique_items'] = unique_items
        
        print(f"Cached statistics for {len(self.user_stats)} active users")
    
    def get_real_users(self, n: int = 100, min_interactions: int = 5) -> List[Dict]:
        """
        Get n random real users with their actual interaction histories.
        
        Args:
            n: Number of users to return
            min_interactions: Minimum interaction count to include user
            
        Returns:
            List of user dictionaries with demographics and real interaction history
        """
        print(f"Selecting {n} real users with at least {min_interactions} interactions...")
        
        # Filter users with sufficient interactions, separating by interaction count
        high_interaction_users = []  # >14 interactions
        low_interaction_users = []   # min_interactions to 14 interactions
        
        for _, user in self.users_df.iterrows():
            user_id = user['user_id']
            if user_id in self.user_stats:
                interaction_count = self.user_stats[user_id]['total_interactions']
                if interaction_count >= min_interactions:
                    if interaction_count > 14:
                        high_interaction_users.append(user)
                    else:
                        low_interaction_users.append(user)
        
        print(f"Found {len(high_interaction_users)} high-interaction users (>14) and {len(low_interaction_users)} low-interaction users ({min_interactions}-14)")
        
        # Ensure more than half have >14 interactions
        min_high_interaction = (n // 2) + 1  # More than 50%
        
        selected_users = []
        
        # First, select from high-interaction users (prioritize these)
        if len(high_interaction_users) >= min_high_interaction:
            selected_high = random.sample(high_interaction_users, min_high_interaction)
        else:
            print(f"Warning: Only {len(high_interaction_users)} high-interaction users available, using all")
            selected_high = high_interaction_users
        
        selected_users.extend(selected_high)
        remaining_slots = n - len(selected_high)
        
        # Fill remaining slots with low-interaction users if available
        if remaining_slots > 0 and len(low_interaction_users) > 0:
            if len(low_interaction_users) >= remaining_slots:
                selected_low = random.sample(low_interaction_users, remaining_slots)
            else:
                selected_low = low_interaction_users
            selected_users.extend(selected_low)
        
        # If we still need more users, add remaining high-interaction users
        remaining_slots = n - len(selected_users)
        if remaining_slots > 0:
            remaining_high = [user for user in high_interaction_users if user not in selected_users]
            if len(remaining_high) >= remaining_slots:
                selected_users.extend(random.sample(remaining_high, remaining_slots))
            else:
                selected_users.extend(remaining_high)
        
        print(f"Selected {len(selected_users)} total users: {len([u for u in selected_users if self.user_stats[u['user_id']]['total_interactions'] > 14])} high-interaction (>14), {len([u for u in selected_users if self.user_stats[u['user_id']]['total_interactions'] <= 14])} low-interaction (≤14)")
        
        if len(selected_users) < n:
            print(f"Warning: Only {len(selected_users)} users available, returning all")
        
        # Build user profiles with real data
        real_user_profiles = []
        for user in selected_users:
            user_id = user['user_id']
            stats = self.user_stats[user_id]
            
            # Determine interaction pattern category
            pattern = self._categorize_interaction_pattern(stats)
            
            profile = {
                'user_id': int(user_id),
                'age': int(user['age']),
                'gender': user['gender'],
                'income': int(user['income']),
                'profession': user.get('profession', 'Other'),
                'location': user.get('location', 'Urban'),
                'education_level': user.get('education_level', 'High School'),
                'marital_status': user.get('marital_status', 'Single'),
                'interaction_history': stats['unique_items'][:50],  # Limit to 50 most recent
                'interaction_stats': {
                    'total_interactions': stats['total_interactions'],
                    'views': stats['views'],
                    'cart_adds': stats['cart'],
                    'purchases': stats['purchase'],
                    'unique_items': len(stats['unique_items'])
                },
                'interaction_pattern': pattern,
                'summary': f"{user['age']}yr {user['gender']}, ${user['income']:,}, {stats['total_interactions']} interactions"
            }
            
            real_user_profiles.append(profile)
        
        # Sort by total interactions (most active first)  
        real_user_profiles.sort(key=lambda x: x['interaction_stats']['total_interactions'], reverse=True)
        
        print(f"Generated {len(real_user_profiles)} real user profiles")
        return real_user_profiles
    
    def _categorize_interaction_pattern(self, stats: Dict) -> str:
        """Categorize user based on their actual interaction pattern."""
        total = stats['total_interactions']
        purchases = stats['purchase']
        
        if total <= 10:
            return "Light Browsing"
        elif total <= 25 and purchases <= 1:
            return "Window Shopping" 
        elif total <= 40 and purchases <= 3:
            return "Serious Shopper"
        elif purchases >= 8:
            return "Frequent Buyer"
        elif total >= 50:
            return "Power User"
        else:
            return "Regular User"
    
    def get_user_interaction_details(self, user_id: int) -> Dict:
        """Get detailed interaction breakdown for a specific user."""
        if user_id not in self.user_stats:
            return {"error": "User not found"}
        
        stats = self.user_stats[user_id]
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        
        # Timeline of interactions with item details
        timeline = []
        for _, interaction in user_interactions.iterrows():
            product_id = interaction['product_id']
            
            # Get item details from items_df if available
            item_info = {'brand': 'Unknown', 'category_code': 'Unknown', 'price': 0.0}
            if self.items_df is not None:
                item_match = self.items_df[self.items_df['product_id'] == product_id]
                if not item_match.empty:
                    item_row = item_match.iloc[0]
                    item_info = {
                        'brand': str(item_row.get('brand', 'Unknown')),
                        'category_code': str(item_row.get('category_code', 'Unknown')),
                        'price': float(item_row.get('price', 0.0))
                    }
            
            timeline.append({
                'timestamp': interaction['event_time'],
                'event_type': interaction['event_type'], 
                'product_id': int(product_id),
                'brand': item_info['brand'],
                'category_code': item_info['category_code'],
                'price': item_info['price']
            })
        
        # Sort by timestamp (most recent first)
        timeline.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'user_id': user_id,
            'total_interactions': stats['total_interactions'],
            'breakdown': {
                'views': stats['views'],
                'cart_adds': stats['cart'], 
                'purchases': stats['purchase']
            },
            'unique_items_count': len(stats['unique_items']),
            'timeline': timeline[:30],  # Last 30 interactions with details
            'interaction_pattern': self._categorize_interaction_pattern(stats)
        }
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics of the entire dataset."""
        total_interactions = len(self.interactions_df)
        total_users = len(self.users_df)
        active_users = len(self.user_stats)
        
        # Event type breakdown
        event_counts = self.interactions_df['event_type'].value_counts().to_dict()
        
        # User demographics summary
        avg_age = self.users_df['age'].mean()
        avg_income = self.users_df['income'].mean() 
        gender_split = self.users_df['gender'].value_counts().to_dict()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_interactions': total_interactions,
            'event_breakdown': event_counts,
            'demographics': {
                'avg_age': round(avg_age, 1),
                'avg_income': round(avg_income, 0),
                'gender_split': gender_split
            }
        }


def main():
    """Demo the real user selector."""
    selector = RealUserSelector()
    
    # Get dataset summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = selector.get_dataset_summary()
    print(f"Total Users: {summary['total_users']:,}")
    print(f"Active Users: {summary['active_users']:,}")
    print(f"Total Interactions: {summary['total_interactions']:,}")
    print(f"Event Breakdown: {summary['event_breakdown']}")
    print(f"Demographics: {summary['demographics']}")
    
    # Get 10 real users for demo
    print("\n" + "="*60)
    print("REAL USER PROFILES (Top 10)")
    print("="*60)
    real_users = selector.get_real_users(n=10, min_interactions=10)
    
    for i, user in enumerate(real_users, 1):
        stats = user['interaction_stats']
        print(f"{i:2d}. User {user['user_id']}: {user['summary']}")
        print(f"    Pattern: {user['interaction_pattern']}")
        print(f"    Actions: {stats['views']} views, {stats['cart_adds']} cart, {stats['purchases']} purchases")
        print(f"    History: {len(user['interaction_history'])} unique items")
        print()
    
    # Show detailed breakdown for first user
    if real_users:
        print("="*60)
        print("DETAILED INTERACTION BREAKDOWN (First User)")
        print("="*60)
        details = selector.get_user_interaction_details(real_users[0]['user_id'])
        print(f"User ID: {details['user_id']}")
        print(f"Total Interactions: {details['total_interactions']}")
        print(f"Pattern: {details['interaction_pattern']}")
        print(f"Breakdown: {details['breakdown']}")
        print(f"Recent Timeline: {len(details['timeline'])} events")
        
        for event in details['timeline'][-5:]:  # Show last 5
            print(f"  - {event['timestamp']}: {event['event_type']} item {event['product_id']}")


if __name__ == "__main__":
    main()