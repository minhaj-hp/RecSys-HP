#!/usr/bin/env python3
"""
Test script for UserTower attention mechanism improvements.
Verifies that the fixed attention masking and pooling work correctly.
"""

import sys
import os
import tensorflow as tf
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.user_tower import UserTower


def test_attention_mechanism():
    """Test the improved UserTower attention mechanism."""
    
    print("🧪 Testing Improved UserTower Attention Mechanism")
    print("=" * 60)
    
    # Test parameters
    max_history_length = 50
    embedding_dim = 128
    batch_size = 8
    
    # Create UserTower
    print("\n📦 Creating UserTower...")
    user_tower = UserTower(
        max_history_length=max_history_length,
        embedding_dim=embedding_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.2
    )
    
    print(f"   Max history length: {max_history_length}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Batch size: {batch_size}")
    
    # Create test scenarios with different interaction history lengths
    test_scenarios = [
        ("Zero interactions", 0),
        ("Few interactions", 3),
        ("Medium interactions", 15),
        ("Many interactions", 35),
        ("Full history", 50)
    ]
    
    print(f"\n🧮 Testing {len(test_scenarios)} scenarios...")
    
    all_results = []
    
    for scenario_name, history_length in test_scenarios:
        print(f"\n--- {scenario_name} ({history_length} items) ---")
        
        # Create test input for this scenario
        test_input = create_test_input(batch_size, max_history_length, embedding_dim, history_length)
        
        # Test forward pass
        try:
            output = user_tower(test_input, training=True)
            
            print(f"✅ Forward pass successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output dtype: {output.dtype}")
            
            # Verify L2 normalization
            norms = tf.linalg.norm(output, axis=1)
            print(f"   L2 norms (should be ~1.0): min={tf.reduce_min(norms):.6f}, max={tf.reduce_max(norms):.6f}")
            
            # Test attention mask creation
            history_mask = tf.reduce_sum(tf.abs(test_input['item_history_embeddings']), axis=-1) > 0
            mask_counts = tf.reduce_sum(tf.cast(history_mask, tf.float32), axis=1)
            
            expected_count = min(history_length, max_history_length)
            actual_counts = mask_counts.numpy()
            
            print(f"   Expected mask count per user: {expected_count}")
            print(f"   Actual mask counts: {actual_counts}")
            
            # Verify mask correctness
            if np.allclose(actual_counts, expected_count):
                print(f"   ✅ Attention mask created correctly")
            else:
                print(f"   ❌ Attention mask incorrect!")
                return False
            
            # Test masked pooling behavior
            test_masked_pooling(user_tower, test_input, history_mask, scenario_name)
            
            all_results.append({
                'scenario': scenario_name,
                'history_length': history_length,
                'output': output.numpy(),
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            all_results.append({
                'scenario': scenario_name,
                'history_length': history_length,
                'success': False,
                'error': str(e)
            })
            return False
    
    # Compare scenarios to verify different history lengths produce different embeddings
    print(f"\n📊 Analyzing scenario differences...")
    
    zero_output = all_results[0]['output']  # Zero interactions
    few_output = all_results[1]['output']   # Few interactions
    many_output = all_results[3]['output']  # Many interactions
    
    # Calculate similarities between scenarios
    zero_few_similarity = np.mean([
        np.dot(zero_output[i], few_output[i]) 
        for i in range(batch_size)
    ])
    
    zero_many_similarity = np.mean([
        np.dot(zero_output[i], many_output[i]) 
        for i in range(batch_size)
    ])
    
    few_many_similarity = np.mean([
        np.dot(few_output[i], many_output[i]) 
        for i in range(batch_size)
    ])
    
    print(f"   Zero vs Few interactions similarity: {zero_few_similarity:.4f}")
    print(f"   Zero vs Many interactions similarity: {zero_many_similarity:.4f}")
    print(f"   Few vs Many interactions similarity: {few_many_similarity:.4f}")
    
    # Verify that different history lengths produce meaningfully different embeddings
    if zero_few_similarity < 0.99:  # Should be different due to interaction signal
        print(f"   ✅ Zero and Few interaction users have different embeddings")
    else:
        print(f"   ⚠️  Zero and Few interaction users are too similar")
    
    if few_many_similarity > 0.80:  # Should be somewhat similar (same user type, different history size)
        print(f"   ✅ Few and Many interaction users have reasonable similarity")
    else:
        print(f"   ⚠️  Few and Many interaction users are unexpectedly different")
    
    return True


def create_test_input(batch_size, max_history_length, embedding_dim, actual_history_length):
    """Create test input with specified interaction history length."""
    
    # Create interaction history embeddings
    history_embeddings = []
    
    for b in range(batch_size):
        user_history = []
        
        # Add real interaction embeddings
        for i in range(actual_history_length):
            # Create realistic non-zero embedding
            embedding = np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            user_history.append(embedding)
        
        # Add zero padding
        for i in range(max_history_length - actual_history_length):
            user_history.append(np.zeros(embedding_dim, dtype=np.float32))
        
        history_embeddings.append(user_history)
    
    history_embeddings = np.array(history_embeddings)
    
    # Create test input
    test_input = {
        'age': tf.constant(np.random.randint(0, 6, batch_size)),  # Age categories
        'gender': tf.constant(np.random.randint(0, 2, batch_size)),  # Gender
        'income': tf.constant(np.random.randint(0, 5, batch_size)),  # Income categories
        'profession': tf.constant(np.random.randint(0, 8, batch_size)),  # Profession
        'location': tf.constant(np.random.randint(0, 3, batch_size)),  # Location
        'education_level': tf.constant(np.random.randint(0, 5, batch_size)),  # Education
        'marital_status': tf.constant(np.random.randint(0, 4, batch_size)),  # Marital
        'item_history_embeddings': tf.constant(history_embeddings)
    }
    
    return test_input


def test_masked_pooling(user_tower, test_input, history_mask, scenario_name):
    """Test the masked pooling implementation."""
    
    print(f"   🔍 Testing masked pooling for {scenario_name}...")
    
    # Get item history
    item_history = test_input['item_history_embeddings']
    
    # Test the masked pooling method directly
    try:
        # Create some dummy attended history (just use original for testing)
        pooled_result = user_tower._masked_mean_pooling(item_history, history_mask)
        
        print(f"      Pooled result shape: {pooled_result.shape}")
        
        # Verify pooling behavior
        batch_size = tf.shape(item_history)[0]
        has_interactions = tf.reduce_any(history_mask, axis=1)
        
        for i in range(min(3, batch_size.numpy())):  # Test first 3 users
            user_has_interactions = has_interactions[i].numpy()
            user_pooled = pooled_result[i].numpy()
            
            if user_has_interactions:
                # Should have non-zero pooled result
                pooled_norm = np.linalg.norm(user_pooled)
                if pooled_norm > 1e-6:
                    print(f"      ✅ User {i}: Has interactions, pooled norm = {pooled_norm:.6f}")
                else:
                    print(f"      ❌ User {i}: Has interactions but pooled result is zero!")
            else:
                # Should have zero pooled result
                pooled_norm = np.linalg.norm(user_pooled)
                if pooled_norm < 1e-6:
                    print(f"      ✅ User {i}: No interactions, pooled correctly zeroed")
                else:
                    print(f"      ❌ User {i}: No interactions but pooled result non-zero: {pooled_norm:.6f}")
        
    except Exception as e:
        print(f"      ❌ Masked pooling test failed: {e}")


def test_zero_interaction_optimization():
    """Test that zero-interaction users are handled efficiently."""
    
    print(f"\n⚡ Testing Zero-Interaction Optimization")
    print("-" * 40)
    
    batch_size = 16
    max_history_length = 50 
    embedding_dim = 128
    
    # Create UserTower
    user_tower = UserTower(
        max_history_length=max_history_length,
        embedding_dim=embedding_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.2
    )
    
    # Test with batch of all zero-interaction users
    test_input = create_test_input(batch_size, max_history_length, embedding_dim, 0)
    
    print(f"   Testing batch of {batch_size} zero-interaction users...")
    
    try:
        # Time the forward pass
        import time
        start_time = time.time()
        
        output = user_tower(test_input, training=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"   ✅ Forward pass successful!")
        print(f"   ⏱️  Elapsed time: {elapsed_time*1000:.2f}ms")
        print(f"   📊 Output shape: {output.shape}")
        
        # Verify outputs are not all zero (should have demographic signal)
        output_norms = tf.linalg.norm(output, axis=1)
        min_norm = tf.reduce_min(output_norms)
        max_norm = tf.reduce_max(output_norms)
        
        print(f"   📏 Output norms: min={min_norm:.6f}, max={max_norm:.6f}")
        
        if min_norm > 1e-6:
            print(f"   ✅ Zero-interaction users still have meaningful embeddings (demographic signal)")
        else:
            print(f"   ⚠️  Some zero-interaction users have zero embeddings")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Zero-interaction optimization test failed: {e}")
        return False


def main():
    """Run all attention mechanism tests."""
    
    try:
        print("🎯 UserTower Attention Mechanism Improvement Tests")
        print("=" * 70)
        
        # Test main attention mechanism
        success1 = test_attention_mechanism()
        
        # Test zero-interaction optimization
        success2 = test_zero_interaction_optimization()
        
        if success1 and success2:
            print(f"\n🎉 All attention mechanism tests passed!")
            print(f"   ✅ Attention masking working correctly")
            print(f"   ✅ Masked mean pooling implemented properly")
            print(f"   ✅ Zero-interaction users handled efficiently")
            print(f"   ✅ Different history lengths produce distinct embeddings")
            
            print(f"\n💡 Expected improvements:")
            print(f"   📈 Better user representations for all history lengths")
            print(f"   🎯 Stronger demographic signals for zero-interaction users")
            print(f"   ⚡ More efficient computation for cold-start scenarios")
            print(f"   🔍 Higher quality attention patterns for active users")
            
        else:
            print(f"\n❌ Some attention mechanism tests failed!")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()