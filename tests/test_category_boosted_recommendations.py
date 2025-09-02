#!/usr/bin/env python3
"""
Comprehensive test script for recommend_items_category_boosted function.

This script tests:
1. Normal operation with interaction history
2. Edge cases (no history, empty categories)
3. Policy compliance (50% category allocation)
4. Output validation and performance
5. Category distribution analysis
"""

import sys
import os
import time
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.recommendation_engine import RecommendationEngine


class CategoryBoostedTester:
    """Comprehensive tester for category-boosted recommendations."""
    
    def __init__(self):
        """Initialize the test environment."""
        print("🚀 Initializing CategoryBoostedTester...")
        try:
            self.engine = RecommendationEngine()
            print("✅ RecommendationEngine loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load RecommendationEngine: {e}")
            sys.exit(1)
            
        # Load items data for category analysis
        try:
            self.items_df = pd.read_csv("datasets/items.csv")
            print(f"✅ Loaded {len(self.items_df)} items for analysis")
        except Exception as e:
            print(f"❌ Failed to load items.csv: {e}")
            sys.exit(1)
    
    def get_item_categories(self, item_ids: List[int]) -> Dict[int, str]:
        """Get category codes for given item IDs."""
        categories = {}
        for item_id in item_ids:
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                categories[item_id] = item_row.iloc[0]['category_code']
            else:
                categories[item_id] = 'unknown'
        return categories
    
    def analyze_category_distribution(self, recommendations: List[Tuple[int, float, Dict]], 
                                    interaction_history: List[int]) -> Dict:
        """Analyze the category distribution of recommendations vs user history."""
        
        # Get user's historical categories (2-level subcategories)
        user_categories = set()
        for item_id in interaction_history:
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                full_category = item_row.iloc[0]['category_code']
                if '.' in full_category:
                    category_parts = full_category.split('.')
                    if len(category_parts) >= 2:
                        subcategory = f"{category_parts[0]}.{category_parts[1]}"
                    else:
                        subcategory = category_parts[0]
                else:
                    subcategory = full_category
                user_categories.add(subcategory)
        
        # Analyze recommendations
        rec_categories = []
        matching_categories = 0
        
        for item_id, score, item_info in recommendations:
            rec_category = item_info.get('category_code', 'unknown')
            
            # Extract 2-level subcategory for matching
            if '.' in rec_category:
                category_parts = rec_category.split('.')
                if len(category_parts) >= 2:
                    rec_subcategory = f"{category_parts[0]}.{category_parts[1]}"
                else:
                    rec_subcategory = category_parts[0]
            else:
                rec_subcategory = rec_category
                
            rec_categories.append(rec_subcategory)
            
            if rec_subcategory in user_categories:
                matching_categories += 1
        
        return {
            'user_categories': user_categories,
            'recommendation_categories': rec_categories,
            'matching_count': matching_categories,
            'total_recommendations': len(recommendations),
            'category_alignment_percentage': (matching_categories / len(recommendations) * 100) if recommendations else 0,
            'category_distribution': Counter(rec_categories)
        }


def test_normal_operation():
    """Test normal operation with interaction history."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Normal Operation with Interaction History")
    print("="*60)
    
    tester = CategoryBoostedTester()
    
    # Test user with diverse interaction history
    test_user = {
        'age': 32,
        'gender': 'male',
        'income': 75000,
        'profession': 'Technology',
        'location': 'Urban',
        'education_level': "Bachelor's",
        'marital_status': 'Married',
        'interaction_history': [1000978, 1001588, 1001618, 1002456, 1003789, 1004123]  # Mix of categories
    }
    
    print(f"👤 Test user: {test_user['age']}-year-old {test_user['gender']}, ${test_user['income']:,} income")
    print(f"📚 Interaction history: {test_user['interaction_history']}")
    
    # Get historical categories
    historical_categories = tester.get_item_categories(test_user['interaction_history'])
    print(f"🏷️  Historical categories: {list(historical_categories.values())}")
    
    start_time = time.time()
    recommendations = tester.engine.recommend_items_category_boosted(
        age=test_user['age'],
        gender=test_user['gender'],
        income=test_user['income'],
        profession=test_user['profession'],
        location=test_user['location'],
        education_level=test_user['education_level'],
        marital_status=test_user['marital_status'],
        interaction_history=test_user['interaction_history'],
        k=10
    )
    execution_time = time.time() - start_time
    
    print(f"\n⏱️  Execution time: {execution_time:.3f} seconds")
    print(f"📊 Generated {len(recommendations)} recommendations")
    
    # Analyze results
    analysis = tester.analyze_category_distribution(recommendations, test_user['interaction_history'])
    
    print(f"\n📈 POLICY COMPLIANCE ANALYSIS:")
    print(f"   Category Alignment: {analysis['category_alignment_percentage']:.1f}%")
    print(f"   Expected: ~50% (policy target)")
    print(f"   Matching items: {analysis['matching_count']}/{analysis['total_recommendations']}")
    
    print(f"\n📋 RECOMMENDATION DETAILS:")
    for i, (item_id, score, info) in enumerate(recommendations[:5], 1):
        category = info.get('category_code', 'unknown')
        brand = info.get('brand', 'unknown')
        price = info.get('price', 0)
        print(f"   {i}. Item {item_id}: {brand} | {category} | ${price:.2f} (Score: {score:.4f})")
    
    # Test passes if we get recommendations and category alignment is reasonable
    success = (len(recommendations) == 10 and 
               40 <= analysis['category_alignment_percentage'] <= 70)  # Allow some variance
    
    print(f"\n✅ Test Result: {'PASS' if success else 'FAIL'}")
    return success


def test_edge_cases():
    """Test edge cases: no history, empty categories."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Edge Cases (No History & Empty Categories)")
    print("="*60)
    
    tester = CategoryBoostedTester()
    success_count = 0
    
    # Test Case 2a: No interaction history
    print("\n📝 Test 2a: No interaction history")
    try:
        recommendations = tester.engine.recommend_items_category_boosted(
            age=25,
            gender='female',
            income=50000,
            interaction_history=None,  # No history
            k=10
        )
        print(f"   Result: {len(recommendations)} recommendations (fallback to collaborative)")
        print(f"   ✅ Handled gracefully with fallback")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Case 2b: Empty interaction history
    print("\n📝 Test 2b: Empty interaction history")
    try:
        recommendations = tester.engine.recommend_items_category_boosted(
            age=25,
            gender='female',
            income=50000,
            interaction_history=[],  # Empty list
            k=10
        )
        print(f"   Result: {len(recommendations)} recommendations (fallback to collaborative)")
        print(f"   ✅ Handled gracefully with fallback")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test Case 2c: Non-existent item IDs in history
    print("\n📝 Test 2c: Non-existent item IDs in history")
    try:
        recommendations = tester.engine.recommend_items_category_boosted(
            age=30,
            gender='male',
            income=60000,
            interaction_history=[99999999, 88888888, 77777777],  # Non-existent IDs
            k=10
        )
        print(f"   Result: {len(recommendations)} recommendations")
        print(f"   ✅ Handled invalid item IDs gracefully")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    success = success_count == 3
    print(f"\n✅ Edge Cases Result: {'PASS' if success else 'FAIL'} ({success_count}/3)")
    return success


def test_policy_compliance():
    """Test strict policy compliance: 50% category allocation."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Policy Compliance (50% Category Allocation)")
    print("="*60)
    
    tester = CategoryBoostedTester()
    test_cases = [
        {
            'name': 'Electronics User',
            'interaction_history': [1000001, 1000002, 1000003, 1000004, 1000005],
            'expected_electronics': True
        },
        {
            'name': 'Books User', 
            'interaction_history': [1000978, 1001588, 1001618],
            'expected_books': True
        },
        {
            'name': 'Mixed Categories User',
            'interaction_history': [1000978, 1002456, 1003789, 1004123, 1005678, 1006789],
            'expected_mixed': True
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📝 Testing: {case['name']}")
        
        try:
            recommendations = tester.engine.recommend_items_category_boosted(
                age=30,
                gender='male',
                income=70000,
                interaction_history=case['interaction_history'],
                k=10
            )
            
            analysis = tester.analyze_category_distribution(recommendations, case['interaction_history'])
            alignment_pct = analysis['category_alignment_percentage']
            
            print(f"   📊 Category Alignment: {alignment_pct:.1f}%")
            print(f"   🎯 Policy Target: 50%")
            print(f"   📈 User Categories: {len(analysis['user_categories'])}")
            print(f"   📋 Recommendation Categories: {dict(analysis['category_distribution'])}")
            
            # Policy compliance check (allow 10% tolerance)
            compliance = 40 <= alignment_pct <= 60
            print(f"   {'✅' if compliance else '❌'} Policy Compliance: {'PASS' if compliance else 'FAIL'}")
            
            results.append(compliance)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append(False)
    
    success = all(results)
    print(f"\n✅ Policy Compliance Result: {'PASS' if success else 'FAIL'} ({sum(results)}/{len(results)})")
    return success


def test_performance_and_validation():
    """Test performance and output validation."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Performance and Output Validation")
    print("="*60)
    
    tester = CategoryBoostedTester()
    
    # Performance test
    print("\n📝 Performance Test (10 users)")
    execution_times = []
    
    test_users = [
        {'age': 25, 'gender': 'male', 'income': 50000, 'history': [1000978, 1001588]},
        {'age': 30, 'gender': 'female', 'income': 60000, 'history': [1002456, 1003789, 1004123]},
        {'age': 35, 'gender': 'male', 'income': 80000, 'history': [1005678, 1006789, 1007890, 1008901]},
        {'age': 28, 'gender': 'female', 'income': 55000, 'history': [1000978, 1002456, 1005678]},
        {'age': 40, 'gender': 'male', 'income': 90000, 'history': [1001588, 1003789, 1006789, 1008901, 1009012]},
    ] * 2  # 10 total users
    
    for i, user in enumerate(test_users):
        start_time = time.time()
        recommendations = tester.engine.recommend_items_category_boosted(
            age=user['age'],
            gender=user['gender'],
            income=user['income'],
            interaction_history=user['history'],
            k=10
        )
        exec_time = time.time() - start_time
        execution_times.append(exec_time)
        
        # Validate output format
        assert len(recommendations) <= 10, f"Too many recommendations: {len(recommendations)}"
        assert all(len(rec) == 3 for rec in recommendations), "Invalid recommendation format"
        assert all(isinstance(rec[0], int) for rec in recommendations), "Invalid item_id type"
        assert all(isinstance(rec[1], float) for rec in recommendations), "Invalid score type"
        assert all(isinstance(rec[2], dict) for rec in recommendations), "Invalid item_info type"
    
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)
    
    print(f"   ⏱️  Average execution time: {avg_time:.3f} seconds")
    print(f"   ⏱️  Maximum execution time: {max_time:.3f} seconds")
    print(f"   🎯 Performance target: < 1.0 seconds")
    
    performance_ok = avg_time < 1.0
    print(f"   {'✅' if performance_ok else '❌'} Performance: {'PASS' if performance_ok else 'FAIL'}")
    
    # Output validation test
    print("\n📝 Output Validation Test")
    sample_recs = tester.engine.recommend_items_category_boosted(
        age=30,
        gender='male',
        income=70000,
        interaction_history=[1000978, 1001588, 1001618],
        k=5
    )
    
    validation_checks = [
        (len(sample_recs) == 5, "Correct number of recommendations"),
        (all('product_id' in rec[2] for rec in sample_recs), "Product ID present"),
        (all('category_code' in rec[2] for rec in sample_recs), "Category code present"),
        (all('brand' in rec[2] for rec in sample_recs), "Brand present"),
        (all('price' in rec[2] for rec in sample_recs), "Price present"),
        (all(0 <= rec[1] <= 1 for rec in sample_recs), "Scores in valid range"),
    ]
    
    validation_success = all(check[0] for check in validation_checks)
    
    for passed, description in validation_checks:
        print(f"   {'✅' if passed else '❌'} {description}")
    
    success = performance_ok and validation_success
    print(f"\n✅ Performance & Validation Result: {'PASS' if success else 'FAIL'}")
    return success


def run_comprehensive_test():
    """Run all tests and generate summary report."""
    print("🎯 CATEGORY BOOSTED RECOMMENDATIONS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Normal Operation", test_normal_operation),
        ("Edge Cases", test_edge_cases),
        ("Policy Compliance", test_policy_compliance),
        ("Performance & Validation", test_performance_and_validation)
    ]
    
    results = []
    
    for test_name, test_function in tests:
        try:
            result = test_function()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Generate summary report
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<10} {test_name}")
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The recommend_items_category_boosted function is working correctly.")
        print("✅ Policy compliance verified: 50% category allocation maintained")
        print("✅ Edge cases handled gracefully")
        print("✅ Performance within acceptable limits")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)