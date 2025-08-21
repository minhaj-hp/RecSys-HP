import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle


class DataProcessor:
    """Handles data loading and preprocessing for the two-tower model."""
    
    def __init__(self, data_path: str = "datasets/"):
        self.data_path = data_path
        self.item_vocab = {}
        self.category_vocab = {}
        self.brand_vocab = {}
        self.user_vocab = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets."""
        items_df = pd.read_csv(f"{self.data_path}/items.csv")
        users_df = pd.read_csv(f"{self.data_path}/users.csv") 
        interactions_df = pd.read_csv(f"{self.data_path}/interactions.csv")
        
        return items_df, users_df, interactions_df
    
    def build_vocabularies(self, items_df: pd.DataFrame, users_df: pd.DataFrame, 
                          interactions_df: pd.DataFrame) -> None:
        """Build vocabulary mappings for categorical features."""
        
        # Item vocabulary
        unique_items = pd.concat([
            items_df['product_id'],
            interactions_df['product_id']
        ]).unique()
        self.item_vocab = {item: idx for idx, item in enumerate(unique_items)}
        
        # Category vocabulary
        unique_categories = items_df['category_id'].unique()
        self.category_vocab = {cat: idx for idx, cat in enumerate(unique_categories)}
        
        # Brand vocabulary (handle missing values)
        unique_brands = items_df['brand'].fillna('unknown').unique()
        self.brand_vocab = {brand: idx for idx, brand in enumerate(unique_brands)}
        
        # User vocabulary
        unique_users = users_df['user_id'].unique()
        self.user_vocab = {user: idx for idx, user in enumerate(unique_users)}
        
        print(f"Vocabularies built:")
        print(f"  Items: {len(self.item_vocab)}")
        print(f"  Categories: {len(self.category_vocab)}")
        print(f"  Brands: {len(self.brand_vocab)}")
        print(f"  Users: {len(self.user_vocab)}")
    
    def prepare_item_features(self, items_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare item features for training."""
        items_df = items_df.fillna({'brand': 'unknown'})
        
        item_features = {
            'product_id': np.array([self.item_vocab.get(item, 0) for item in items_df['product_id']]),
            'category_id': np.array([self.category_vocab.get(cat, 0) for cat in items_df['category_id']]),
            'brand_id': np.array([self.brand_vocab.get(brand, 0) for brand in items_df['brand']]),
            'price': items_df['price'].values.astype(np.float32)
        }
        
        return item_features
    
    def create_user_interaction_history(self, 
                                       interactions_df: pd.DataFrame,
                                       items_df: pd.DataFrame,
                                       max_history_length: int = 50) -> Dict[int, List[int]]:
        """Create user interaction histories sorted by timestamp."""
        
        # Convert timestamp to datetime with timezone handling
        interactions_df = interactions_df.copy()
        interactions_df['event_time'] = pd.to_datetime(interactions_df['event_time'], utc=True)
        
        # Sort by user and timestamp
        interactions_sorted = interactions_df.sort_values(['user_id', 'event_time'])
        
        # Build user histories
        user_histories = defaultdict(list)
        for _, row in interactions_sorted.iterrows():
            user_id = row['user_id']
            item_id = self.item_vocab.get(row['product_id'], 0)
            user_histories[user_id].append(item_id)
        
        # Limit history length
        for user_id in user_histories:
            if len(user_histories[user_id]) > max_history_length:
                user_histories[user_id] = user_histories[user_id][-max_history_length:]
        
        return dict(user_histories)
    
    def create_positive_negative_pairs(self, 
                                     interactions_df: pd.DataFrame,
                                     items_df: pd.DataFrame,
                                     negative_samples_per_positive: int = 4) -> pd.DataFrame:
        """Create positive and negative user-item pairs for training."""
        
        # Get all unique items for negative sampling
        all_items = set(self.item_vocab.keys())
        
        # Create positive pairs
        positive_pairs = []
        for _, row in interactions_df.iterrows():
            if row['user_id'] in self.user_vocab and row['product_id'] in self.item_vocab:
                positive_pairs.append({
                    'user_id': row['user_id'],
                    'product_id': row['product_id'],
                    'rating': 1.0  # Implicit positive feedback
                })
        
        # Create negative pairs
        negative_pairs = []
        user_item_interactions = set(
            (row['user_id'], row['product_id']) 
            for _, row in interactions_df.iterrows()
        )
        
        for pos_pair in positive_pairs:
            user_id = pos_pair['user_id']
            user_interactions = set(
                row['product_id'] for _, row in interactions_df.iterrows() 
                if row['user_id'] == user_id
            )
            
            # Sample negative items
            negative_items = all_items - user_interactions
            if len(negative_items) >= negative_samples_per_positive:
                sampled_negatives = np.random.choice(
                    list(negative_items), 
                    size=negative_samples_per_positive, 
                    replace=False
                )
                
                for neg_item in sampled_negatives:
                    negative_pairs.append({
                        'user_id': user_id,
                        'product_id': neg_item,
                        'rating': 0.0  # Negative feedback
                    })
        
        # Combine positive and negative pairs
        all_pairs = positive_pairs + negative_pairs
        return pd.DataFrame(all_pairs)
    
    def save_vocabularies(self, save_path: str = "src/artifacts/"):
        """Save vocabularies for later use."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        vocab_data = {
            'item_vocab': self.item_vocab,
            'category_vocab': self.category_vocab,
            'brand_vocab': self.brand_vocab,
            'user_vocab': self.user_vocab
        }
        
        with open(f"{save_path}/vocabularies.pkl", 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Vocabularies saved to {save_path}/vocabularies.pkl")
    
    def load_vocabularies(self, load_path: str = "src/artifacts/vocabularies.pkl"):
        """Load vocabularies from file."""
        with open(load_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.item_vocab = vocab_data['item_vocab']
        self.category_vocab = vocab_data['category_vocab']
        self.brand_vocab = vocab_data['brand_vocab']
        self.user_vocab = vocab_data['user_vocab']
        
        print("Vocabularies loaded successfully")


def create_tf_dataset(features: Dict[str, np.ndarray], batch_size: int = 256) -> tf.data.Dataset:
    """Create TensorFlow dataset from features."""
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset