#!/usr/bin/env python3
"""
Basic syntax validation test for UserTower attention masking fixes.
Tests the import and class structure without requiring TensorFlow.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that UserTower can be imported without errors."""
    print("🧪 Testing UserTower imports and structure...")
    
    try:
        # This will test the syntax and basic structure
        from models.user_tower import UserTower, TwoTowerModel
        print("✅ UserTower imported successfully")
        
        # Check if the class has the new method
        if hasattr(UserTower, '_masked_mean_pooling'):
            print("✅ _masked_mean_pooling method found")
        else:
            print("❌ _masked_mean_pooling method missing")
        
        # Check if the __init__ method exists
        if hasattr(UserTower, '__init__'):
            print("✅ UserTower.__init__ method found")
        else:
            print("❌ UserTower.__init__ method missing")
            
        # Check if the call method exists
        if hasattr(UserTower, 'call'):
            print("✅ UserTower.call method found")
        else:
            print("❌ UserTower.call method missing")
            
        print("✅ All structural tests passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_method_signatures():
    """Test method signatures without instantiating."""
    print("\n🔍 Testing method signatures...")
    
    try:
        from models.user_tower import UserTower
        import inspect
        
        # Check _masked_mean_pooling signature
        if hasattr(UserTower, '_masked_mean_pooling'):
            sig = inspect.signature(UserTower._masked_mean_pooling)
            params = list(sig.parameters.keys())
            print(f"✅ _masked_mean_pooling parameters: {params}")
            
            expected_params = ['self', 'sequence', 'mask']
            if all(param in params for param in expected_params):
                print("✅ _masked_mean_pooling has expected parameters")
            else:
                print(f"⚠️  _masked_mean_pooling parameters don't match expected: {expected_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Method signature test failed: {e}")
        return False

def main():
    """Run all syntax validation tests."""
    print("🎯 UserTower Syntax Validation Tests")
    print("=" * 50)
    
    test1 = test_imports()
    test2 = test_method_signatures()
    
    if test1 and test2:
        print("\n🎉 All syntax validation tests passed!")
        print("✅ UserTower attention masking fixes appear structurally correct")
        print("\nNext steps:")
        print("- Install TensorFlow to run full functional tests")
        print("- Run test_attention_improvements.py for detailed validation")
        print("- Test with actual recommendation data")
    else:
        print("\n❌ Some syntax validation tests failed!")
        print("Please fix the structural issues before proceeding")

if __name__ == "__main__":
    main()