#!/usr/bin/env python3
"""
Basic test to check data loading and preprocessing.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from src.preprocessing.data_loader import DataProcessor

def test_data_loading():
    """Test basic data loading functionality."""
    print("Testing data loading...")
    
    try:
        # Initialize data processor
        processor = DataProcessor("datasets/")
        
        # Load data
        items_df, users_df, interactions_df = processor.load_data()
        
        print(f"✅ Data loaded successfully:")
        print(f"  - Items: {len(items_df)} records")
        print(f"  - Users: {len(users_df)} records")
        print(f"  - Interactions: {len(interactions_df)} records")
        
        # Check data types
        print(f"\n✅ Data types:")
        print(f"  - Items price dtype: {items_df['price'].dtype}")
        print(f"  - Users age dtype: {users_df['age'].dtype}")
        print(f"  - Users income dtype: {users_df['income'].dtype}")
        
        # Build vocabularies
        processor.build_vocabularies(items_df, users_df, interactions_df)
        
        # Test item features preparation
        item_features = processor.prepare_item_features(items_df)
        
        print(f"\n✅ Item features prepared:")
        for key, arr in item_features.items():
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Test price normalization
        print(f"\n✅ Testing price normalization...")
        prices = np.array(item_features['price']).reshape(-1, 1)
        print(f"  - Price array shape: {prices.shape}")
        print(f"  - Price range: {prices.min():.2f} - {prices.max():.2f}")
        
        import tensorflow as tf
        normalizer = tf.keras.layers.Normalization()
        normalizer.adapt(prices)
        print(f"  - Normalization layer adapted successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 All basic tests passed! Ready for training.")
    else:
        print("\n💥 Tests failed. Please check the errors above.")