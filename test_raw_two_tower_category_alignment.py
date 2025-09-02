#!/usr/bin/env python3
"""
Raw Two-Tower Category Alignment Test

Tests how well the raw two-tower retrieval method preserves category alignment
between user interaction history and recommendations, specifically examining
the 1.3x category boost effectiveness.
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Set

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference.recommendation_engine import RecommendationEngine
    print("✅ Successfully imported RecommendationEngine")
except Exception as e:
    print(f"❌ Failed to import RecommendationEngine: {e}")
    sys.exit(1)


class RawTwoTowerCategoryTester:
    """Test category alignment in raw two-tower retrieval."""
    
    def __init__(self):
        print("🔧 Initializing Raw Two-Tower Category Alignment Tester...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            raise
    
    def create_test_users_with_history(self) -> List[Dict]:
        """Create test users with specific interaction histories to test category boosting."""
        
        return [
            {
                'name': 'ElectronicsEnthusiast',
                'age': 28,
                'gender': 'male',
                'income': 75000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Single',
                'interaction_history': [1000978, 1001588, 1001618, 1002000, 1002827, 1002225, 7005176, 4000088],  # Electronics items
                'expected_categories': ['electronics']
            },
            {
                'name': 'FashionLover',
                'age': 26,
                'gender': 'female',
                'income': 60000,
                'profession': 'Marketing',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Single',
                'interaction_history': [28721300, 28720710, 28717990, 28722200],  # Fashion/apparel items
                'expected_categories': ['apparel']
            },
            {
                'name': 'HealthcareProfessional',
                'age': 35,
                'gender': 'female',
                'income': 68000,
                'profession': 'Healthcare',
                'location': 'Suburban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': [1003304, 1003311, 1002525, 4800369, 2403019],  # Healthcare/professional items
                'expected_categories': ['appliances', 'electronics']
            },
            {
                'name': 'HomeMaker',
                'age': 42,
                'gender': 'female',
                'income': 45000,
                'profession': 'Other',
                'location': 'Suburban',
                'education_level': "High School",
                'marital_status': 'Married',
                'interaction_history': [4800369, 2403019, 28720710, 28717990],  # Home/kitchen items
                'expected_categories': ['appliances', 'apparel']
            },
            {
                'name': 'YoungStudent',
                'age': 20,
                'gender': 'male',
                'income': 15000,
                'profession': 'Other',
                'location': 'Urban',
                'education_level': "Some College",
                'marital_status': 'Single',
                'interaction_history': [1004615, 1004613, 1004701, 1000978],  # Student/budget items
                'expected_categories': ['electronics']
            }
        ]
    
    def analyze_interaction_categories(self, user: Dict) -> Dict[str, int]:
        """Analyze the categories in a user's interaction history."""
        
        category_counts = {}
        
        for item_id in user['interaction_history']:
            item_row = self.engine.items_df[self.engine.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                category = item_row.iloc[0]['category_code']
                if pd.notna(category):
                    # Extract top-level category
                    top_category = category.split('.')[0]
                    category_counts[top_category] = category_counts.get(top_category, 0) + 1
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    def analyze_recommendation_categories(self, recommendations: List[Tuple]) -> Dict[str, int]:
        """Analyze the categories in recommendations."""
        
        category_counts = {}
        
        for item_id, score, item_info in recommendations:
            category = item_info.get('category_code', 'Unknown')
            if category and category != 'Unknown':
                # Extract top-level category
                top_category = category.split('.')[0]
                category_counts[top_category] = category_counts.get(top_category, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    def calculate_category_alignment_score(self, user_categories: Dict[str, int], rec_categories: Dict[str, int]) -> float:
        """Calculate alignment score between user history and recommendations."""
        
        if not user_categories or not rec_categories:
            return 0.0
        
        # Calculate overlap at top-level categories
        user_top_categories = set(cat for cat in user_categories.keys() if '.' not in cat)
        rec_top_categories = set(cat for cat in rec_categories.keys() if '.' not in cat)
        
        if not user_top_categories:
            return 0.0
        
        overlap = len(user_top_categories.intersection(rec_top_categories))
        return overlap / len(user_top_categories)
    
    def test_category_boosting_effect(self, user: Dict) -> Dict:
        """Test the effect of category boosting on recommendations."""
        
        print(f"\n🧪 Testing Category Boosting for {user['name']}...")
        print(f"   Demographics: {user['age']}y {user['gender']} {user['profession']} (${user['income']:,})")
        print(f"   Interaction History: {len(user['interaction_history'])} items")
        
        # Get user's historical categories
        user_categories = self.analyze_interaction_categories(user)
        print(f"   Historical Categories: {dict(list(user_categories.items())[:5])}...")
        
        # Get recommendations with default category boosting (1.3x)
        try:
            recommendations = self.engine.recommend_items_raw_two_tower(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                profession=user['profession'],
                location=user['location'],
                education_level=user['education_level'],
                marital_status=user['marital_status'],
                interaction_history=user['interaction_history'],
                k=20,
                exclude_history=True,
                category_boost=1.3  # Default boosting
            )
            
            if not recommendations:
                print("   ❌ No recommendations returned")
                return {'error': 'No recommendations'}
            
            print(f"   ✅ Got {len(recommendations)} recommendations")
            
            # Analyze recommendation categories
            rec_categories = self.analyze_recommendation_categories(recommendations)
            
            # Calculate alignment score
            alignment_score = self.calculate_category_alignment_score(user_categories, rec_categories)
            
            # Show top recommendation details
            print(f"   🏆 Top 3 recommendations:")
            for i, (item_id, score, item_info) in enumerate(recommendations[:3]):
                category = item_info.get('category_code', 'Unknown')
                brand = item_info.get('brand', 'Unknown')
                price = item_info.get('price', 0)
                
                # Check if this category was in user's history
                is_historical = any(cat in user_categories for cat in [category, category.split('.')[0]] if cat)
                boost_indicator = "🔥" if is_historical else "🆕"
                
                print(f"      {i+1}. {boost_indicator} Item {item_id} - {brand} - {category} - ${price:.2f} (score: {score:.4f})")
            
            # Show category breakdown
            print(f"   📊 Recommendation Categories:")
            for category, count in sorted(rec_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                if '.' not in category:  # Show only top-level categories
                    is_historical = category in user_categories
                    boost_indicator = "🔥" if is_historical else "🆕"
                    percentage = (count / len(recommendations)) * 100
                    print(f"      {boost_indicator} {category}: {count} items ({percentage:.1f}%)")
            
            # Calculate boosting effectiveness
            boosted_count = 0
            total_score_boost = 0
            
            for item_id, score, item_info in recommendations:
                category = item_info.get('category_code', '')
                if category:
                    top_category = category.split('.')[0]
                    if category in user_categories or top_category in user_categories:
                        boosted_count += 1
                        # Estimate original score before boosting
                        original_score = score / 1.3
                        boost_effect = score - original_score
                        total_score_boost += boost_effect
            
            boost_effectiveness = (boosted_count / len(recommendations)) * 100
            
            print(f"   ⚡ Category Boosting Analysis:")
            print(f"      Boosted Items: {boosted_count}/{len(recommendations)} ({boost_effectiveness:.1f}%)")
            print(f"      Alignment Score: {alignment_score:.3f} (0=no match, 1=perfect match)")
            print(f"      Average Score Boost: {total_score_boost/max(boosted_count, 1):.4f}")
            
            return {
                'user': user['name'],
                'recommendations': len(recommendations),
                'user_categories': user_categories,
                'rec_categories': rec_categories,
                'alignment_score': alignment_score,
                'boosted_count': boosted_count,
                'boost_effectiveness': boost_effectiveness,
                'top_recommendations': recommendations[:5]
            }
            
        except Exception as e:
            print(f"   ❌ Error getting recommendations: {e}")
            return {'error': str(e)}
    
    def compare_with_without_boosting(self, user: Dict) -> Dict:
        """Compare recommendations with and without category boosting."""
        
        print(f"\n🔄 Comparing Boosting Effects for {user['name']}...")
        
        try:
            # Get recommendations WITHOUT boosting (category_boost=1.0)
            recs_no_boost = self.engine.recommend_items_raw_two_tower(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                profession=user['profession'],
                location=user['location'],
                education_level=user['education_level'],
                marital_status=user['marital_status'],
                interaction_history=user['interaction_history'],
                k=10,
                exclude_history=True,
                category_boost=1.0  # No boosting
            )
            
            # Get recommendations WITH boosting (category_boost=1.3)
            recs_with_boost = self.engine.recommend_items_raw_two_tower(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                profession=user['profession'],
                location=user['location'],
                education_level=user['education_level'],
                marital_status=user['marital_status'],
                interaction_history=user['interaction_history'],
                k=10,
                exclude_history=True,
                category_boost=1.3  # Default boosting
            )
            
            if not recs_no_boost or not recs_with_boost:
                print("   ❌ Failed to get both recommendation sets")
                return {'error': 'Missing recommendations'}
            
            # Analyze differences
            no_boost_items = [item_id for item_id, _, _ in recs_no_boost]
            with_boost_items = [item_id for item_id, _, _ in recs_with_boost]
            
            overlap = len(set(no_boost_items).intersection(set(with_boost_items)))
            overlap_percentage = (overlap / len(no_boost_items)) * 100
            
            print(f"   📊 Recommendation Overlap: {overlap}/{len(no_boost_items)} items ({overlap_percentage:.1f}%)")
            
            # Show items that changed due to boosting
            boosted_up = [item for item in with_boost_items if item not in no_boost_items]
            boosted_out = [item for item in no_boost_items if item not in with_boost_items]
            
            print(f"   ⬆️  Items boosted INTO top-10: {len(boosted_up)}")
            print(f"   ⬇️  Items boosted OUT of top-10: {len(boosted_out)}")
            
            if boosted_up:
                print(f"   🔥 Items promoted by category boosting:")
                for item_id in boosted_up[:3]:
                    item_info = next((info for iid, _, info in recs_with_boost if iid == item_id), {})
                    category = item_info.get('category_code', 'Unknown')
                    brand = item_info.get('brand', 'Unknown')
                    print(f"      - Item {item_id}: {brand} - {category}")
            
            return {
                'overlap_percentage': overlap_percentage,
                'boosted_up_count': len(boosted_up),
                'boosted_out_count': len(boosted_out),
                'boosted_up_items': boosted_up[:3],
                'significant_change': overlap_percentage < 80
            }
            
        except Exception as e:
            print(f"   ❌ Error in boost comparison: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_test(self):
        """Run comprehensive category alignment testing."""
        
        print("🚀 Starting Raw Two-Tower Category Alignment Testing")
        print("="*70)
        
        test_users = self.create_test_users_with_history()
        results = []
        
        # Test category boosting effectiveness
        for user in test_users:
            result = self.test_category_boosting_effect(user)
            if 'error' not in result:
                results.append(result)
            
            # Also test boost comparison for first few users
            if len(results) <= 2:
                self.compare_with_without_boosting(user)
        
        # Summary analysis
        print("\n" + "="*70)
        print("📈 SUMMARY: Category Alignment Analysis")
        print("="*70)
        
        if results:
            avg_alignment = np.mean([r['alignment_score'] for r in results])
            avg_boost_effectiveness = np.mean([r['boost_effectiveness'] for r in results])
            avg_boosted_count = np.mean([r['boosted_count'] for r in results])
            
            print(f"📊 Overall Performance:")
            print(f"   Average Category Alignment: {avg_alignment:.3f} (0=no match, 1=perfect)")
            print(f"   Average Boost Effectiveness: {avg_boost_effectiveness:.1f}% of recommendations")
            print(f"   Average Boosted Items per User: {avg_boosted_count:.1f} items")
            
            print(f"\n🎯 Assessment:")
            if avg_alignment > 0.6:
                print(f"   ✅ EXCELLENT: Category alignment > 60% - boosting working well!")
            elif avg_alignment > 0.4:
                print(f"   ⚠️  MODERATE: Category alignment 40-60% - room for improvement")
            else:
                print(f"   ❌ POOR: Category alignment < 40% - boosting may not be effective")
            
            if avg_boost_effectiveness > 50:
                print(f"   ✅ HIGH IMPACT: {avg_boost_effectiveness:.1f}% of recommendations are boosted")
            else:
                print(f"   ⚠️  LOW IMPACT: Only {avg_boost_effectiveness:.1f}% of recommendations are boosted")
        
        else:
            print("❌ No successful test results to analyze")
        
        print("\n🎉 Category Alignment Testing Completed!")


def main():
    """Run the category alignment tests."""
    
    try:
        tester = RawTwoTowerCategoryTester()
        tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()