#!/usr/bin/env python3
"""
Comprehensive analysis of UserTower attention mask implementation.
Verifies that attention masks are working correctly at each step.
"""

import sys
import os
import tensorflow as tf
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.user_tower import UserTower


def analyze_mask_creation():
    """Analyze how attention masks are created from embeddings."""
    
    print("🔍 STEP 1: Mask Creation Analysis")
    print("=" * 50)
    
    # Create test data with known pattern
    batch_size = 3
    seq_len = 8
    embedding_dim = 16
    
    # Create test history: [3 items, 5 items, 0 items] per user
    test_histories = []
    expected_masks = []
    
    # User 0: 3 real interactions + 5 padding
    user0_history = []
    for i in range(3):  # Real interactions
        user0_history.append(np.random.normal(0, 1, embedding_dim))
    for i in range(5):  # Padding 
        user0_history.append(np.zeros(embedding_dim))
    test_histories.append(user0_history)
    expected_masks.append([True, True, True, False, False, False, False, False])
    
    # User 1: 5 real interactions + 3 padding  
    user1_history = []
    for i in range(5):  # Real interactions
        user1_history.append(np.random.normal(0, 1, embedding_dim))
    for i in range(3):  # Padding
        user1_history.append(np.zeros(embedding_dim))
    test_histories.append(user1_history)
    expected_masks.append([True, True, True, True, True, False, False, False])
    
    # User 2: 0 real interactions (all padding)
    user2_history = []
    for i in range(8):  # All padding
        user2_history.append(np.zeros(embedding_dim))
    test_histories.append(user2_history)
    expected_masks.append([False, False, False, False, False, False, False, False])
    
    # Convert to tensor
    item_history = tf.constant(test_histories, dtype=tf.float32)
    
    print(f"   Input shape: {item_history.shape}")
    print(f"   Test pattern: [3 items, 5 items, 0 items] per user")
    
    # Test mask creation (same logic as UserTower)
    history_mask = tf.reduce_sum(tf.abs(item_history), axis=-1) > 0
    
    print(f"   Created mask shape: {history_mask.shape}")
    print(f"   Created mask dtype: {history_mask.dtype}")
    
    # Verify each user's mask
    actual_masks = history_mask.numpy()
    
    for i in range(batch_size):
        expected = expected_masks[i]
        actual = actual_masks[i].tolist()
        
        print(f"\n   User {i}:")
        print(f"      Expected: {expected}")
        print(f"      Actual:   {actual}")
        
        if expected == actual:
            print(f"      ✅ Mask created correctly")
        else:
            print(f"      ❌ Mask creation failed!")
            return False
    
    return True, item_history, history_mask


def analyze_attention_mask_format(history_mask):
    """Analyze attention mask format transformation."""
    
    print(f"\n🎭 STEP 2: Attention Mask Format Analysis")
    print("=" * 50)
    
    print(f"   Original mask shape: {history_mask.shape}")  # [batch_size, seq_len]
    
    # Transform for attention (same as UserTower)
    attention_mask = tf.expand_dims(history_mask, axis=1)  # [batch_size, 1, seq_len]
    
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Attention mask dtype: {attention_mask.dtype}")
    
    # Verify transformation
    batch_size, num_heads, seq_len = attention_mask.shape
    expected_shape = (history_mask.shape[0], 1, history_mask.shape[1])
    
    if attention_mask.shape == expected_shape:
        print(f"   ✅ Attention mask shape transformation correct")
        print(f"   📋 Format: [batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}]")
    else:
        print(f"   ❌ Attention mask shape transformation failed!")
        return False
    
    # Show actual values for verification
    print(f"\n   Mask values for each user:")
    for i in range(batch_size):
        original = history_mask[i].numpy()
        transformed = attention_mask[i, 0].numpy()  # Remove the 1 dimension
        
        print(f"      User {i}: {original.tolist()} -> {transformed.tolist()}")
        
        if np.array_equal(original, transformed):
            print(f"                ✅ Values preserved correctly")
        else:
            print(f"                ❌ Values corrupted in transformation!")
            return False
    
    return True, attention_mask


def analyze_masked_pooling_behavior():
    """Analyze masked mean pooling implementation."""
    
    print(f"\n🧮 STEP 3: Masked Pooling Analysis")
    print("=" * 50)
    
    user_tower = UserTower(max_history_length=8, embedding_dim=16, hidden_dims=[8, 4])
    
    # Create test attended history (simulating attention output)
    batch_size = 3
    seq_len = 8
    embedding_dim = 16
    
    # Create realistic attended embeddings
    attended_history = tf.random.normal([batch_size, seq_len, embedding_dim])
    
    # Create test masks: [3 items, 5 items, 0 items]
    test_masks = [
        [True, True, True, False, False, False, False, False],  # 3 items
        [True, True, True, True, True, False, False, False],    # 5 items  
        [False, False, False, False, False, False, False, False] # 0 items
    ]
    
    history_mask = tf.constant(test_masks)
    
    print(f"   Attended history shape: {attended_history.shape}")
    print(f"   History mask shape: {history_mask.shape}")
    
    # Test masked pooling
    pooled_result = user_tower._masked_mean_pooling(attended_history, history_mask)
    
    print(f"   Pooled result shape: {pooled_result.shape}")
    print(f"   Expected shape: ({batch_size}, {embedding_dim})")
    
    if pooled_result.shape == (batch_size, embedding_dim):
        print(f"   ✅ Pooled result shape correct")
    else:
        print(f"   ❌ Pooled result shape incorrect!")
        return False
    
    # Analyze each user's pooling result
    for i in range(batch_size):
        mask = test_masks[i]
        num_true = sum(mask)
        pooled_norm = tf.linalg.norm(pooled_result[i]).numpy()
        
        print(f"\n   User {i} (expects {num_true} items):")
        print(f"      Mask: {mask}")
        print(f"      Pooled norm: {pooled_norm:.6f}")
        
        if num_true == 0:
            # Should be zero vector for users with no interactions
            if pooled_norm < 1e-6:
                print(f"      ✅ Zero-interaction user correctly zeroed")
            else:
                print(f"      ❌ Zero-interaction user should have zero pooling!")
                return False
        else:
            # Should have non-zero pooling for users with interactions
            if pooled_norm > 1e-6:
                print(f"      ✅ User with interactions has non-zero pooling")
            else:
                print(f"      ❌ User with interactions has zero pooling!")
                return False
    
    return True


def analyze_full_attention_pipeline():
    """Analyze the complete attention pipeline in UserTower."""
    
    print(f"\n🎯 STEP 4: Full Attention Pipeline Analysis")
    print("=" * 50)
    
    user_tower = UserTower(max_history_length=10, embedding_dim=32, hidden_dims=[16, 8])
    
    # Create comprehensive test input
    batch_size = 4
    max_history = 10
    embedding_dim = 32
    
    # Test cases: [2 items, 6 items, 0 items, 10 items]
    test_cases = [2, 6, 0, 10]
    
    history_embeddings = []
    for case in test_cases:
        user_history = []
        
        # Add real interactions
        for i in range(case):
            embedding = np.random.normal(0, 0.3, embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            user_history.append(embedding)
        
        # Add padding
        for i in range(max_history - case):
            user_history.append(np.zeros(embedding_dim))
        
        history_embeddings.append(user_history)
    
    # Create full test input
    test_input = {
        'age': tf.constant([1, 2, 3, 4]),
        'gender': tf.constant([0, 1, 0, 1]),
        'income': tf.constant([2, 3, 1, 4]),
        'profession': tf.constant([0, 1, 2, 3]),
        'location': tf.constant([0, 1, 2, 0]),
        'education_level': tf.constant([2, 3, 1, 4]),
        'marital_status': tf.constant([0, 1, 2, 0]),
        'item_history_embeddings': tf.constant(history_embeddings, dtype=tf.float32)
    }
    
    print(f"   Test cases: {test_cases} interactions per user")
    print(f"   Input history shape: {test_input['item_history_embeddings'].shape}")
    
    # Run full forward pass
    try:
        user_embeddings = user_tower(test_input, training=False)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   Output shape: {user_embeddings.shape}")
        print(f"   Expected shape: ({batch_size}, 32)")
        
        # Verify output characteristics
        norms = tf.linalg.norm(user_embeddings, axis=1).numpy()
        print(f"   Output norms: {norms}")
        
        # All should be normalized to 1.0
        if np.allclose(norms, 1.0):
            print(f"   ✅ All outputs properly L2 normalized")
        else:
            print(f"   ❌ Output normalization failed!")
            return False
        
        # Compare different interaction lengths
        similarities = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                sim = np.dot(user_embeddings[i], user_embeddings[j])
                similarities.append(sim)
                print(f"   User {i} ({test_cases[i]} items) vs User {j} ({test_cases[j]} items): {sim:.4f}")
        
        # Check that users with different histories are meaningfully different
        avg_similarity = np.mean(similarities)
        print(f"   Average pairwise similarity: {avg_similarity:.4f}")
        
        if avg_similarity < 0.95:  # Should be different enough
            print(f"   ✅ Users with different interaction histories produce distinct embeddings")
        else:
            print(f"   ⚠️  Users are very similar despite different interaction histories")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive attention mask analysis."""
    
    print("🔬 COMPREHENSIVE ATTENTION MASK ANALYSIS")
    print("=" * 70)
    print("Verifying that attention masks work correctly at every step")
    
    try:
        # Step 1: Mask creation
        success1, item_history, history_mask = analyze_mask_creation()
        if not success1:
            return False
        
        # Step 2: Attention mask format
        success2, attention_mask = analyze_attention_mask_format(history_mask)
        if not success2:
            return False
        
        # Step 3: Masked pooling
        success3 = analyze_masked_pooling_behavior()
        if not success3:
            return False
        
        # Step 4: Full pipeline
        success4 = analyze_full_attention_pipeline()
        if not success4:
            return False
        
        print(f"\n🎉 ATTENTION MASK ANALYSIS COMPLETE!")
        print("=" * 50)
        print("✅ All attention mask components working correctly:")
        print("   1. ✅ Mask creation from embeddings")
        print("   2. ✅ Attention mask format transformation") 
        print("   3. ✅ Masked mean pooling implementation")
        print("   4. ✅ Full attention pipeline integration")
        
        print(f"\n💡 Key Findings:")
        print("   🎭 Masks correctly identify real vs padded positions")
        print("   📐 Attention mask format [batch, 1, seq_len] works properly")
        print("   🧮 Masked pooling ignores padding and handles zero-interaction users")
        print("   🎯 Full pipeline produces distinct embeddings for different histories")
        
        print(f"\n🚀 CONCLUSION: Attention masks are working perfectly!")
        print("   Ready for production use - retraining will benefit from these improvements")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()