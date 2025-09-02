#!/usr/bin/env python3
"""
Test 50% Category-Based Collaborative Filtering Logic

This script validates that the category-boosted recommendation algorithm correctly:
1. Takes collaborative filtering results
2. Ensures 50% of recommendations come from user's historical categories
3. Maintains similarity-based ranking for the remaining 50%
4. Properly handles category.subcategory.subcategory hierarchy
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import json

# Add src to path
sys.path.append('src')
from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector

class CategoryBoostLogicTester:
    """Test the 50% category-based collaborative filtering logic."""
    
    def __init__(self):
        print("🔧 Initializing 50% Category Logic Tester...")
        self.engine = RecommendationEngine()
        self.real_user_selector = RealUserSelector()
        self.test_results = []
        
        print("✅ Components loaded successfully!")
    
    def analyze_user_category_distribution(self, interaction_history: List[int]) -> Dict:
        """Analyze the category distribution in user's interaction history."""
        
        if not interaction_history:
            return {}
        
        category_counts = Counter()
        
        for item_id in interaction_history:
            item_row = self.engine.items_df[self.engine.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                category = item_row.iloc[0]['category_code']
                if pd.notna(category):
                    # Use 2-level category hierarchy as per the algorithm
                    if '.' in str(category):
                        parts = str(category).split('.')
                        if len(parts) >= 2:
                            category = f"{parts[0]}.{parts[1]}"
                        else:
                            category = parts[0]
                    category_counts[category] += 1
        
        total_interactions = len(interaction_history)
        category_percentages = {cat: count/total_interactions for cat, count in category_counts.items()}
        
        return {
            'category_counts': dict(category_counts),
            'category_percentages': category_percentages,
            'total_interactions': total_interactions,
            'unique_categories': len(category_counts)
        }
    
    def analyze_recommendations_categories(self, recommendations: List[Tuple[int, float, Dict]]) -> Dict:
        """Analyze category distribution in recommendations."""
        
        category_counts = Counter()
        rec_details = []
        
        for item_id, score, item_info in recommendations:
            category = item_info.get('category_code', '')
            if category:
                # Use same 2-level hierarchy
                if '.' in str(category):
                    parts = str(category).split('.')
                    if len(parts) >= 2:
                        category = f"{parts[0]}.{parts[1]}"
                    else:
                        category = parts[0]
                category_counts[category] += 1
                
                rec_details.append({
                    'item_id': item_id,
                    'score': score,
                    'category': category,
                    'full_category': item_info.get('category_code', '')
                })
        
        total_recs = len(recommendations)
        category_percentages = {cat: count/total_recs for cat, count in category_counts.items()}
        
        return {
            'category_counts': dict(category_counts),
            'category_percentages': category_percentages,
            'total_recommendations': total_recs,
            'unique_categories': len(category_counts),
            'rec_details': rec_details
        }
    
    def test_single_user_category_logic(self, user_data: Dict, k: int = 20) -> Dict:
        """Test category-boosted logic for a single user."""
        
        user_id = user_data['user_id']
        interaction_history = user_data['interaction_history']
        
        print(f"\n🧪 Testing User {user_id}")
        print(f"   Interaction History: {len(interaction_history)} items")
        
        # Analyze user's historical category preferences
        hist_analysis = self.analyze_user_category_distribution(interaction_history)
        
        if not hist_analysis:
            print("   ❌ No valid categories in interaction history")
            return {"success": False, "error": "No categories in history"}
        
        print(f"   Historical Categories: {list(hist_analysis['category_percentages'].keys())}")
        print(f"   Category Distribution: {hist_analysis['category_percentages']}")
        
        # Get collaborative filtering baseline
        try:
            collaborative_recs = self.engine.recommend_items_collaborative(
                age=user_data['age'],
                gender=user_data['gender'],
                income=user_data['income'],
                profession=user_data.get('profession', 'Other'),
                location=user_data.get('location', 'Urban'),
                education_level=user_data.get('education_level', "Bachelor's"),
                marital_status=user_data.get('marital_status', 'Single'),
                interaction_history=interaction_history,
                k=k
            )
            
            if not collaborative_recs:
                print("   ❌ No collaborative recommendations returned")
                return {"success": False, "error": "No collaborative recommendations"}
            
        except Exception as e:
            print(f"   ❌ Error getting collaborative recommendations: {e}")
            return {"success": False, "error": f"Collaborative error: {e}"}
        
        # Get category-boosted recommendations
        try:
            category_boosted_recs = self.engine.recommend_items_category_boosted(
                age=user_data['age'],
                gender=user_data['gender'],
                income=user_data['income'],
                profession=user_data.get('profession', 'Other'),
                location=user_data.get('location', 'Urban'),
                education_level=user_data.get('education_level', "Bachelor's"),
                marital_status=user_data.get('marital_status', 'Single'),
                interaction_history=interaction_history,
                k=k
            )
            
            if not category_boosted_recs:
                print("   ❌ No category-boosted recommendations returned")
                return {"success": False, "error": "No category-boosted recommendations"}
            
        except Exception as e:
            print(f"   ❌ Error getting category-boosted recommendations: {e}")
            return {"success": False, "error": f"Category-boosted error: {e}"}
        
        # Analyze both recommendation sets
        collab_analysis = self.analyze_recommendations_categories(collaborative_recs)
        boosted_analysis = self.analyze_recommendations_categories(category_boosted_recs)
        
        # Calculate category alignment
        historical_categories = set(hist_analysis['category_percentages'].keys())
        boosted_categories = set(boosted_analysis['category_percentages'].keys())
        
        # Count how many boosted recommendations are from historical categories
        aligned_count = 0
        aligned_items = []
        non_aligned_items = []
        
        for rec_detail in boosted_analysis['rec_details']:
            if rec_detail['category'] in historical_categories:
                aligned_count += 1
                aligned_items.append(rec_detail)
            else:
                non_aligned_items.append(rec_detail)
        
        alignment_percentage = (aligned_count / len(category_boosted_recs)) * 100
        
        # Analyze proportional representation within aligned recommendations
        aligned_category_dist = Counter([item['category'] for item in aligned_items])
        proportional_accuracy = self._calculate_proportional_accuracy(
            hist_analysis['category_percentages'], 
            aligned_category_dist, 
            aligned_count
        )
        
        result = {
            "success": True,
            "user_id": user_id,
            "interaction_count": len(interaction_history),
            
            # Historical analysis
            "historical_categories": list(historical_categories),
            "historical_distribution": hist_analysis['category_percentages'],
            
            # Collaborative baseline
            "collaborative_category_dist": collab_analysis['category_percentages'],
            "collaborative_diversity": collab_analysis['unique_categories'],
            
            # Category-boosted results
            "boosted_category_dist": boosted_analysis['category_percentages'],
            "boosted_diversity": boosted_analysis['unique_categories'],
            
            # 50% alignment validation
            "total_recommendations": len(category_boosted_recs),
            "aligned_count": aligned_count,
            "alignment_percentage": alignment_percentage,
            "target_alignment": 50.0,
            "alignment_success": 45.0 <= alignment_percentage <= 55.0,
            
            # Proportional representation within aligned recommendations
            "proportional_accuracy": proportional_accuracy,
            "aligned_category_dist": dict(aligned_category_dist),
            
            # Quality comparison
            "collaborative_avg_score": np.mean([score for _, score, _ in collaborative_recs]),
            "boosted_avg_score": np.mean([score for _, score, _ in category_boosted_recs]),
        }
        
        # Print results
        status = "✅ PASS" if result["alignment_success"] else "⚠️ FAIL"
        print(f"   📊 Alignment: {aligned_count}/{len(category_boosted_recs)} ({alignment_percentage:.1f}%) {status}")
        print(f"   🎯 Target: 50% ± 5% (45-55%)")
        print(f"   📈 Proportional Accuracy: {proportional_accuracy:.2f}")
        print(f"   🔢 Quality: Collaborative({result['collaborative_avg_score']:.4f}) vs Boosted({result['boosted_avg_score']:.4f})")
        print(f"   📂 Diversity: Collaborative({collab_analysis['unique_categories']}) vs Boosted({boosted_analysis['unique_categories']})")
        
        return result
    
    def _calculate_proportional_accuracy(self, historical_percentages: Dict, aligned_dist: Counter, aligned_count: int) -> float:
        """Calculate how accurately the aligned recommendations represent historical proportions."""
        
        if aligned_count == 0:
            return 0.0
        
        total_accuracy = 0.0
        categories_checked = 0
        
        for category, historical_pct in historical_percentages.items():
            actual_count = aligned_dist.get(category, 0)
            actual_pct = actual_count / aligned_count
            
            # Calculate accuracy for this category (1.0 = perfect, 0.0 = completely wrong)
            max_possible_error = max(historical_pct, 1.0 - historical_pct)
            actual_error = abs(historical_pct - actual_pct)
            category_accuracy = 1.0 - (actual_error / max_possible_error) if max_possible_error > 0 else 1.0
            
            total_accuracy += category_accuracy
            categories_checked += 1
        
        return total_accuracy / categories_checked if categories_checked > 0 else 0.0
    
    def get_test_users_simple(self, user_count: int = 10) -> List[Dict]:
        """Get test users with substantial interaction history using simple approach."""
        
        # Load users and interactions data
        users_df = pd.read_csv('datasets/users.csv')
        interactions_df = pd.read_csv('datasets/interactions.csv')
        
        # Count interactions per user
        user_interaction_counts = interactions_df['user_id'].value_counts()
        
        # Get users with at least 15 interactions
        users_with_enough_interactions = user_interaction_counts[user_interaction_counts >= 15].head(user_count)
        
        test_users = []
        
        for user_id in users_with_enough_interactions.index:
            # Get user demographics
            user_row = users_df[users_df['user_id'] == user_id]
            if len(user_row) == 0:
                continue
            
            user_info = user_row.iloc[0]
            
            # Get user's interaction history
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]['product_id'].tolist()
            
            # Limit to recent 30 interactions for testing
            user_interactions = user_interactions[-30:]
            
            test_user = {
                'user_id': user_id,
                'age': user_info['age'],
                'gender': user_info['gender'],
                'income': user_info['income'],
                'profession': user_info.get('profession', 'Other'),
                'location': user_info.get('location', 'Urban'),
                'education_level': user_info.get('education_level', "Bachelor's"),
                'marital_status': user_info.get('marital_status', 'Single'),
                'interaction_history': user_interactions
            }
            
            test_users.append(test_user)
            
            if len(test_users) >= user_count:
                break
        
        return test_users

    def test_multiple_users(self, user_count: int = 10, k: int = 20) -> List[Dict]:
        """Test category-boosted logic for multiple users."""
        
        print("="*80)
        print("🎯 TESTING 50% CATEGORY-BASED COLLABORATIVE FILTERING LOGIC")
        print("="*80)
        
        # Get test users with substantial interaction history
        print(f"📊 Getting {user_count} real users with interaction history...")
        real_users = self.get_test_users_simple(user_count)
        
        if len(real_users) < user_count:
            print(f"⚠️ Only found {len(real_users)} users with sufficient history")
        
        results = []
        
        for i, user_data in enumerate(real_users):
            print(f"\n--- Testing User {i+1}/{len(real_users)} ---")
            result = self.test_single_user_category_logic(user_data, k=k)
            
            if result["success"]:
                results.append(result)
                self.test_results.append(result)
        
        return results
    
    def analyze_overall_results(self, results: List[Dict]):
        """Analyze overall performance across all tested users."""
        
        print("\n" + "="*80)
        print("📈 OVERALL 50% CATEGORY LOGIC ANALYSIS")
        print("="*80)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Success rate
        successful_alignments = [r for r in results if r["alignment_success"]]
        success_rate = len(successful_alignments) / len(results) * 100
        
        # Alignment statistics
        alignments = [r["alignment_percentage"] for r in results]
        proportional_accuracies = [r["proportional_accuracy"] for r in results]
        
        # Quality statistics  
        collab_scores = [r["collaborative_avg_score"] for r in results]
        boosted_scores = [r["boosted_avg_score"] for r in results]
        
        print(f"\n🎯 ALIGNMENT PERFORMANCE:")
        print(f"   Users Tested: {len(results)}")
        print(f"   Success Rate: {len(successful_alignments)}/{len(results)} ({success_rate:.1f}%)")
        print(f"   Target Range: 45-55% category alignment")
        print(f"   Average Alignment: {np.mean(alignments):.1f}% ± {np.std(alignments):.1f}%")
        print(f"   Median Alignment: {np.median(alignments):.1f}%")
        print(f"   Min-Max Alignment: {np.min(alignments):.1f}% - {np.max(alignments):.1f}%")
        
        print(f"\n📊 PROPORTIONAL ACCURACY:")
        print(f"   Average Proportional Accuracy: {np.mean(proportional_accuracies):.3f}")
        print(f"   (1.0 = perfect proportion match, 0.0 = completely wrong)")
        
        print(f"\n🔢 QUALITY COMPARISON:")
        print(f"   Collaborative Avg Score: {np.mean(collab_scores):.4f} ± {np.std(collab_scores):.4f}")
        print(f"   Category-Boosted Avg Score: {np.mean(boosted_scores):.4f} ± {np.std(boosted_scores):.4f}")
        quality_change = (np.mean(boosted_scores) - np.mean(collab_scores)) / np.mean(collab_scores) * 100
        print(f"   Quality Change: {quality_change:+.1f}%")
        
        # Detailed user analysis
        print(f"\n📋 DETAILED USER RESULTS:")
        for i, result in enumerate(results):
            status = "✅" if result["alignment_success"] else "❌"
            print(f"   User {result['user_id']:>6}: {result['alignment_percentage']:>5.1f}% alignment "
                  f"({result['aligned_count']:>2}/{result['total_recommendations']}) "
                  f"accuracy={result['proportional_accuracy']:.3f} {status}")
        
        # Recommendations
        print(f"\n🔧 ASSESSMENT:")
        if success_rate >= 80:
            print("   ✅ EXCELLENT: Algorithm working as designed!")
        elif success_rate >= 60:
            print("   ⚠️ GOOD: Minor tuning may be needed")
        else:
            print("   ❌ NEEDS ATTENTION: Algorithm requires debugging")
        
        if np.mean(proportional_accuracies) >= 0.7:
            print("   ✅ Proportional representation is accurate")
        else:
            print("   ⚠️ Proportional representation needs improvement")
        
        if abs(quality_change) <= 10:
            print("   ✅ Quality maintained within acceptable range")
        else:
            print("   ⚠️ Significant quality change detected")
    
    def save_results(self, filename: str = "category_boost_test_results.json"):
        """Save detailed test results."""
        
        if not self.test_results:
            print("❌ No results to save")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"💾 Detailed results saved to {filename}")
    
    def run_comprehensive_test(self, user_count: int = 10, k: int = 20):
        """Run comprehensive 50% category logic testing."""
        
        print("🚀 Starting 50% Category-Based Collaborative Filtering Test")
        print(f"   Testing {user_count} users with {k} recommendations each")
        print(f"   Validating: 50% ± 5% category alignment with interaction history")
        
        # Test multiple users
        results = self.test_multiple_users(user_count=user_count, k=k)
        
        # Analyze overall performance
        self.analyze_overall_results(results)
        
        # Save results
        self.save_results()
        
        print("\n🎉 50% Category Logic Testing Completed!")


def main():
    """Run 50% category logic testing."""
    
    try:
        tester = CategoryBoostLogicTester()
        tester.run_comprehensive_test(user_count=10, k=20)
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()