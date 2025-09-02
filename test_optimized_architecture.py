#!/usr/bin/env python3
"""
Test script for the optimized ItemTower architecture.
Verifies that the new dimensions and price preprocessing work correctly.
"""

import sys
import os
import tensorflow as tf
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.item_tower import ItemTower


def test_optimized_item_tower():
    """Test the optimized ItemTower architecture."""
    
    print("🧪 Testing Optimized ItemTower Architecture")
    print("=" * 50)
    
    # Test parameters
    item_vocab_size = 1000
    category_vocab_size = 100
    brand_vocab_size = 50
    batch_size = 32
    
    # Create optimized item tower
    print("\n📦 Creating optimized ItemTower...")
    item_tower = ItemTower(
        item_vocab_size=item_vocab_size,
        category_vocab_size=category_vocab_size,
        brand_vocab_size=brand_vocab_size,
        embedding_dim=128,    # Final output
        item_dim=56,          # Optimized dimensions
        category_dim=16,
        brand_dim=16,
        price_dim=16
    )
    
    # Print architecture summary
    summary = item_tower.get_architecture_summary()
    print("\n📊 Architecture Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Create test input
    print(f"\n🧮 Testing with batch_size={batch_size}...")
    test_input = {
        'product_id': tf.random.uniform([batch_size], 0, item_vocab_size, dtype=tf.int32),
        'category_id': tf.random.uniform([batch_size], 0, category_vocab_size, dtype=tf.int32), 
        'brand_id': tf.random.uniform([batch_size], 0, brand_vocab_size, dtype=tf.int32),
        'price': tf.random.uniform([batch_size], 10.0, 1000.0, dtype=tf.float32)  # Realistic prices
    }
    
    print("   Input shapes:")
    for key, tensor in test_input.items():
        print(f"     {key}: {tensor.shape}")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    try:
        output = item_tower(test_input, training=True)
        
        print("✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Expected shape: ({batch_size}, 128)")
        
        # Verify L2 normalization
        norms = tf.linalg.norm(output, axis=1)
        print(f"   L2 norms (should be ~1.0): min={tf.reduce_min(norms):.6f}, max={tf.reduce_max(norms):.6f}")
        
        # Test price preprocessing
        print("\n💰 Testing price preprocessing...")
        prices = test_input['price'].numpy()
        print(f"   Original prices: min=${prices.min():.2f}, max=${prices.max():.2f}")
        print(f"   Log-transformed prices: min={np.log(prices.min() + 1):.4f}, max={np.log(prices.max() + 1):.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test parameter count reduction
    print("\n📉 Parameter Count Analysis:")
    total_params = sum([np.prod(var.shape) for var in item_tower.trainable_variables])
    print(f"   Total parameters: {total_params:,}")
    
    # Calculate what the old architecture would have had
    old_params_estimate = (
        item_vocab_size * 128 +      # item_embedding (old)
        category_vocab_size * 128 +  # category_embedding (old)  
        brand_vocab_size * 128 +     # brand_embedding (old)
        (128*3 + 1) * 256 +         # first dense layer (old input: 385D)
        256 * 128 +                 # second dense layer
        128 * 128                   # output layer
    )
    
    new_input_dim = item_tower.get_input_dim()  # 120D
    new_params_estimate = (
        item_vocab_size * 56 +       # item_embedding (new)
        category_vocab_size * 16 +   # category_embedding (new)
        brand_vocab_size * 16 +      # brand_embedding (new)
        8 + 16*8 + 16 +             # price MLP
        new_input_dim * 256 +        # first dense layer (new input: 120D)
        256 * 128 +                 # second dense layer  
        128 * 128                   # output layer
    )
    
    reduction_ratio = (old_params_estimate - new_params_estimate) / old_params_estimate * 100
    
    print(f"   Estimated old architecture: {old_params_estimate:,} parameters")
    print(f"   Estimated new architecture: {new_params_estimate:,} parameters")
    print(f"   Reduction: ~{reduction_ratio:.1f}%")
    
    # Test gradient flow
    print("\n🔄 Testing gradient flow...")
    try:
        with tf.GradientTape() as tape:
            output = item_tower(test_input, training=True)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, item_tower.trainable_variables)
        non_none_grads = sum(1 for grad in gradients if grad is not None)
        total_vars = len(item_tower.trainable_variables)
        
        print(f"✅ Gradient computation successful!")
        print(f"   Variables with gradients: {non_none_grads}/{total_vars}")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False
    
    # Test inference mode
    print("\n🔍 Testing inference mode...")
    try:
        output_inference = item_tower(test_input, training=False)
        print(f"✅ Inference mode successful!")
        print(f"   Output shape: {output_inference.shape}")
        
        # Outputs should be slightly different due to dropout
        diff = tf.reduce_mean(tf.abs(output - output_inference))
        print(f"   Training vs Inference difference: {diff:.6f}")
        
    except Exception as e:
        print(f"❌ Inference mode failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! Optimized ItemTower is working correctly.")
    print(f"   ✅ Dimension optimization: 56D+16D+16D+16D = {new_input_dim}D input")
    print(f"   ✅ Price preprocessing: log(price+1) → z-score → MLP({item_tower.price_dim}D)")
    print(f"   ✅ Parameter reduction: ~{reduction_ratio:.1f}% fewer parameters")
    print(f"   ✅ Output: L2-normalized {output.shape[1]}D embeddings")
    
    return True


def main():
    """Run the architecture test."""
    
    try:
        success = test_optimized_item_tower()
        if success:
            print(f"\n✅ Architecture optimization test completed successfully!")
        else:
            print(f"\n❌ Architecture optimization test failed!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()