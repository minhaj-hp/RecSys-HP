import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from datetime import datetime
import pickle
import os

from src.preprocessing.data_loader import DataProcessor


class UserDatasetCreator:
    """Creates user training dataset with demographics and aggregated item embeddings."""
    
    def __init__(self, max_history_length: int = 50):
        self.max_history_length = max_history_length
        self.data_processor = DataProcessor()
        
    def load_item_embeddings(self, embeddings_path: str = "src/artifacts/item_embeddings.npy") -> Dict[int, np.ndarray]:
        """Load pre-trained item embeddings."""
        try:
            return np.load(embeddings_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Warning: {embeddings_path} not found. Creating dummy embeddings...")
            # Create dummy embeddings for demo purposes
            from src.preprocessing.data_loader import DataProcessor
            processor = DataProcessor()
            items_df, users_df, interactions_df = processor.load_data()
            
            dummy_embeddings = {}
            for item_id in items_df['product_id'].unique():
                dummy_embeddings[item_id] = np.random.rand(64).astype(np.float32)
            
            print(f"Created dummy embeddings for {len(dummy_embeddings)} items")
            return dummy_embeddings
    
    def aggregate_user_history_embeddings(self, 
                                        user_histories: Dict[int, List[int]],
                                        item_embeddings: Dict[int, np.ndarray],
                                        embedding_dim: int = 64) -> Dict[int, np.ndarray]:
        """Aggregate item embeddings for each user's interaction history."""
        
        user_aggregated_embeddings = {}
        
        for user_id, item_history in user_histories.items():
            if not item_history:
                # No history - use zero embedding
                user_aggregated_embeddings[user_id] = np.zeros((self.max_history_length, embedding_dim))
                continue
            
            # Get embeddings for items in history
            history_embeddings = []
            for item_idx in item_history:
                if item_idx in item_embeddings:
                    history_embeddings.append(item_embeddings[item_idx])
                else:
                    # Use zero embedding for unknown items
                    history_embeddings.append(np.zeros(embedding_dim))
            
            history_embeddings = np.array(history_embeddings)
            
            # Pad or truncate to max_history_length
            if len(history_embeddings) < self.max_history_length:
                # Pad with zeros
                padding = np.zeros((self.max_history_length - len(history_embeddings), embedding_dim))
                history_embeddings = np.vstack([padding, history_embeddings])
            else:
                # Take most recent interactions
                history_embeddings = history_embeddings[-self.max_history_length:]
            
            user_aggregated_embeddings[user_id] = history_embeddings
        
        return user_aggregated_embeddings
    
    def create_temporal_split(self, 
                            interactions_df: pd.DataFrame,
                            split_date: str = "2019-11-15") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions temporally for training and validation."""
        
        # Convert to datetime and handle timezone issues
        interactions_df = interactions_df.copy()
        interactions_df['event_time'] = pd.to_datetime(interactions_df['event_time'], utc=True)
        split_timestamp = pd.to_datetime(split_date, utc=True)
        
        train_interactions = interactions_df[interactions_df['event_time'] < split_timestamp]
        val_interactions = interactions_df[interactions_df['event_time'] >= split_timestamp]
        
        print(f"Temporal split:")
        print(f"  Training interactions: {len(train_interactions)} (before {split_date})")
        print(f"  Validation interactions: {len(val_interactions)} (after {split_date})")
        
        return train_interactions, val_interactions
    
    def prepare_user_features(self, 
                            users_df: pd.DataFrame,
                            user_aggregated_embeddings: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare user features combining demographics and history embeddings."""
        
        # Filter users that have both demographics and history
        valid_users = set(users_df['user_id']) & set(user_aggregated_embeddings.keys())
        valid_users = sorted(list(valid_users))
        
        # Prepare demographic features
        user_demographics = users_df[users_df['user_id'].isin(valid_users)].copy()
        user_demographics = user_demographics.sort_values('user_id')
        
        # Convert gender to numeric (0=female, 1=male)
        user_demographics['gender_numeric'] = (user_demographics['gender'] == 'male').astype(int)
        
        # Create mapping from user_id to array index
        user_id_to_index = {uid: idx for idx, uid in enumerate(user_demographics['user_id'])}
        
        # Prepare features
        user_features = {
            'user_id_to_index': user_id_to_index,  # Add mapping for later use
            'user_ids': user_demographics['user_id'].values,  # Keep original user IDs
            'age': user_demographics['age'].values.astype(np.float32),
            'gender': user_demographics['gender_numeric'].values.astype(np.int32),
            'income': user_demographics['income'].values.astype(np.float32),
            'item_history_embeddings': np.array([
                user_aggregated_embeddings[uid] for uid in user_demographics['user_id']
            ]).astype(np.float32)
        }
        
        print(f"Prepared user features for {len(valid_users)} users")
        print(f"History embeddings shape: {user_features['item_history_embeddings'].shape}")
        
        return user_features
    
    def create_training_dataset(self, 
                              interactions_df: pd.DataFrame,
                              items_df: pd.DataFrame,
                              users_df: pd.DataFrame,
                              item_embeddings: Dict[int, np.ndarray],
                              negative_samples_per_positive: int = 4) -> Dict[str, np.ndarray]:
        """Create complete training dataset."""
        
        # Load vocabularies
        self.data_processor.build_vocabularies(items_df, users_df, interactions_df)
        
        # Create user histories up to each interaction point
        print("Creating user interaction histories...")
        user_histories = self.data_processor.create_user_interaction_history(
            interactions_df, items_df, self.max_history_length
        )
        
        # Aggregate user history embeddings
        print("Aggregating user history embeddings...")
        user_aggregated_embeddings = self.aggregate_user_history_embeddings(
            user_histories, item_embeddings
        )
        
        # Create positive/negative pairs
        print("Creating positive/negative pairs...")
        training_pairs = self.data_processor.create_positive_negative_pairs(
            interactions_df, items_df, negative_samples_per_positive
        )
        
        # Prepare user features
        user_features = self.prepare_user_features(users_df, user_aggregated_embeddings)
        
        # Prepare item features
        item_features = self.data_processor.prepare_item_features(items_df)
        
        # Create aligned dataset
        print("Creating aligned training dataset...")
        
        # Get valid user-item pairs
        valid_pairs = []
        for _, row in training_pairs.iterrows():
            user_id = row['user_id']
            item_id = row['product_id']
            rating = row['rating']
            
            if (user_id in self.data_processor.user_vocab and 
                item_id in self.data_processor.item_vocab):
                valid_pairs.append({
                    'user_id': user_id,
                    'product_id': item_id,
                    'rating': rating
                })
        
        valid_pairs_df = pd.DataFrame(valid_pairs)
        
        # Create feature arrays for training
        training_features = {}
        
        # User features for each pair - use correct mapping
        user_indices = []
        valid_user_pairs = []
        
        for _, row in valid_pairs_df.iterrows():
            user_id = row['user_id']
            if user_id in user_features['user_id_to_index']:
                user_indices.append(user_features['user_id_to_index'][user_id])
                valid_user_pairs.append(row)
        
        # Filter valid pairs to only those with user features
        valid_pairs_df = pd.DataFrame(valid_user_pairs)
        
        if len(valid_pairs_df) == 0:
            print("Warning: No valid user-item pairs found!")
            return {}
        
        # Now use the correct indices
        training_features['age'] = user_features['age'][user_indices]
        training_features['gender'] = user_features['gender'][user_indices]
        training_features['income'] = user_features['income'][user_indices]
        training_features['item_history_embeddings'] = user_features['item_history_embeddings'][user_indices]
        
        # Item features for each pair
        item_indices = [self.data_processor.item_vocab[iid] for iid in valid_pairs_df['product_id']]
        training_features['product_id'] = item_features['product_id'][item_indices]
        training_features['category_id'] = item_features['category_id'][item_indices]
        training_features['brand_id'] = item_features['brand_id'][item_indices]
        training_features['price'] = item_features['price'][item_indices]
        
        # Ratings
        training_features['rating'] = valid_pairs_df['rating'].values.astype(np.float32)
        
        print(f"Created training dataset with {len(valid_pairs)} samples")
        
        return training_features
    
    def save_dataset(self, 
                    training_features: Dict[str, np.ndarray],
                    save_path: str = "src/artifacts/"):
        """Save the prepared training dataset."""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save features
        with open(f"{save_path}/training_features.pkl", 'wb') as f:
            pickle.dump(training_features, f)
        
        # Save dataset statistics
        stats = {
            'num_samples': len(training_features['rating']),
            'num_positive': np.sum(training_features['rating'] > 0.5),
            'num_negative': np.sum(training_features['rating'] <= 0.5),
            'history_length': training_features['item_history_embeddings'].shape[1],
            'embedding_dim': training_features['item_history_embeddings'].shape[2]
        }
        
        with open(f"{save_path}/dataset_stats.txt", 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Training dataset saved to {save_path}")
        print(f"Dataset statistics: {stats}")
    
    def load_dataset(self, load_path: str = "src/artifacts/training_features.pkl") -> Dict[str, np.ndarray]:
        """Load saved training dataset."""
        with open(load_path, 'rb') as f:
            training_features = pickle.load(f)
        
        print(f"Loaded training dataset with {len(training_features['rating'])} samples")
        return training_features


def main():
    """Main function for user dataset creation."""
    
    # Initialize dataset creator
    dataset_creator = UserDatasetCreator(max_history_length=50)
    
    # Load data
    print("Loading data...")
    data_processor = DataProcessor()
    items_df, users_df, interactions_df = data_processor.load_data()
    
    # Load pre-trained item embeddings
    print("Loading item embeddings...")
    item_embeddings = dataset_creator.load_item_embeddings()
    
    # Sample data for faster processing during development
    print("Sampling data for faster processing...")
    sample_users = users_df.sample(n=min(1000, len(users_df)))
    user_ids = set(sample_users['user_id'])
    
    # Filter interactions to sample users
    sample_interactions = interactions_df[interactions_df['user_id'].isin(user_ids)]
    
    # Filter items to those in sample interactions
    item_ids = set(sample_interactions['product_id'])
    sample_items = items_df[items_df['product_id'].isin(item_ids)]
    
    print(f"Sampled: {len(sample_items)} items, {len(sample_users)} users, {len(sample_interactions)} interactions")
    
    # Create temporal split
    print("Creating temporal split...")
    train_interactions, val_interactions = dataset_creator.create_temporal_split(sample_interactions)
    
    # Create training dataset
    print("Creating training dataset...")
    training_features = dataset_creator.create_training_dataset(
        train_interactions, sample_items, sample_users, item_embeddings,
        negative_samples_per_positive=2  # Reduce for faster processing
    )
    
    # Save dataset
    print("Saving training dataset...")
    dataset_creator.save_dataset(training_features)
    
    # Create validation dataset (smaller sample)
    print("Creating validation dataset...")
    val_sample_size = min(1000, len(val_interactions))
    val_sample = val_interactions.sample(val_sample_size) if val_sample_size > 0 else val_interactions
    
    val_training_features = dataset_creator.create_training_dataset(
        val_sample, sample_items, sample_users, item_embeddings,
        negative_samples_per_positive=1  # Even smaller for validation
    )
    
    # Save validation dataset
    with open("src/artifacts/validation_features.pkl", 'wb') as f:
        pickle.dump(val_training_features, f)
    
    print("User dataset creation completed!")


if __name__ == "__main__":
    main()