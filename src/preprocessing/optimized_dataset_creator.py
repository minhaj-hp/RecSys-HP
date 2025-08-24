"""
Optimized dataset creation script with performance improvements.
"""
import time
import numpy as np
from src.preprocessing.user_data_preparation import UserDatasetCreator
from src.preprocessing.data_loader import DataProcessor, create_tf_dataset


def create_optimized_dataset(max_history_length: int = 50, 
                           batch_size: int = 512,
                           negative_samples_per_positive: int = 2,
                           use_sample: bool = False,
                           sample_size: int = 10000):
    """
    Create dataset with optimized performance settings.
    
    Args:
        max_history_length: Maximum user interaction history length
        batch_size: Batch size for TensorFlow dataset
        negative_samples_per_positive: Negative sampling ratio
        use_sample: Whether to use a sample of the data for faster processing
        sample_size: Size of sample if use_sample=True
    """
    print("Starting optimized dataset creation...")
    start_time = time.time()
    
    # Initialize with optimized settings
    dataset_creator = UserDatasetCreator(max_history_length=max_history_length)
    data_processor = DataProcessor()
    
    # Load data
    print("Loading data...")
    load_start = time.time()
    items_df, users_df, interactions_df = data_processor.load_data()
    print(f"Data loaded in {time.time() - load_start:.2f} seconds")
    
    # Optional: Use sample for faster development/testing
    if use_sample:
        print(f"Using sample of {sample_size} interactions for faster processing...")
        sample_interactions = interactions_df.sample(min(sample_size, len(interactions_df)))
        user_ids = set(sample_interactions['user_id'])
        item_ids = set(sample_interactions['product_id'])
        
        users_df = users_df[users_df['user_id'].isin(user_ids)]
        items_df = items_df[items_df['product_id'].isin(item_ids)]
        interactions_df = sample_interactions
        
        print(f"Sample: {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
    
    # Load embeddings with caching
    print("Loading item embeddings...")
    embed_start = time.time()
    item_embeddings = dataset_creator.load_item_embeddings()
    print(f"Embeddings loaded in {time.time() - embed_start:.2f} seconds")
    
    # Create temporal split
    print("Creating temporal split...")
    split_start = time.time()
    train_interactions, val_interactions = dataset_creator.create_temporal_split(interactions_df)
    print(f"Temporal split created in {time.time() - split_start:.2f} seconds")
    
    # Create training dataset with optimizations
    print("Creating optimized training dataset...")
    train_start = time.time()
    training_features = dataset_creator.create_training_dataset(
        train_interactions, items_df, users_df, item_embeddings,
        negative_samples_per_positive=negative_samples_per_positive
    )
    print(f"Training dataset created in {time.time() - train_start:.2f} seconds")
    
    # Create TensorFlow dataset optimized for CPU
    print("Creating TensorFlow dataset...")
    tf_start = time.time()
    tf_dataset = create_tf_dataset(training_features, batch_size=batch_size)
    print(f"TensorFlow dataset created in {time.time() - tf_start:.2f} seconds")
    
    # Save optimized dataset
    print("Saving dataset...")
    save_start = time.time()
    dataset_creator.save_dataset(training_features, "src/artifacts/")
    
    # Save vocabularies for later use
    data_processor.save_vocabularies("src/artifacts/")
    print(f"Dataset saved in {time.time() - save_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nOptimized dataset creation completed in {total_time:.2f} seconds!")
    print(f"Training samples: {len(training_features['rating'])}")
    print(f"Memory usage optimized for CPU training")
    
    return tf_dataset, training_features


if __name__ == "__main__":
    # Run with optimized settings
    tf_dataset, features = create_optimized_dataset(
        max_history_length=30,  # Reduced for speed
        batch_size=512,         # Larger batches for CPU efficiency
        negative_samples_per_positive=2,  # Reduced sampling ratio
        use_sample=True,        # Use sample for development
        sample_size=50000       # Reasonable sample size
    )
    
    print("\nDataset creation optimization complete!")
    print("Key optimizations applied:")
    print("- Vectorized DataFrame operations")
    print("- Parallel negative sampling")
    print("- Memory-efficient embedding lookup")
    print("- Optimized TensorFlow dataset pipeline")
    print("- LRU caching for embeddings")