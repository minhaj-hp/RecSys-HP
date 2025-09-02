#!/usr/bin/env python3
"""
Test the updated 1.6x category boost in raw two-tower retrieval.
Compare the effectiveness of 1.6x vs the previous 1.3x boost.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference.recommendation_engine import RecommendationEngine
    print("✅ Successfully imported RecommendationEngine")
except Exception as e:
    print(f"❌ Failed to import RecommendationEngine: {e}")
    sys.exit(1)


class CategoryBoostTester:
    """Test the updated 1.6x category boost."""
    
    def __init__(self):
        print("🔧 Initializing Category Boost Tester...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            raise
    
    def test_category_boost_comparison(self):
        """Compare different category boost values."""
        
        # Test user with clear category preferences
        test_user = {
            'name': 'ElectronicsEnthusiast',
            'age': 28,
            'gender': 'male',
            'income': 75000,
            'profession': 'Technology',
            'location': 'Urban',
            'education_level': "Bachelor's",
            'marital_status': 'Single',
            'interaction_history': [1000978, 1001588, 1001618, 1002000, 1002827, 1002225, 7005176, 4000088],  # Electronics items
        }
        
        print(f"\n🧪 Testing Category Boost Values for {test_user['name']}...")
        print(f"   Demographics: {test_user['age']}y {test_user['gender']} {test_user['profession']} (${test_user['income']:,})")
        print(f"   Interaction History: {len(test_user['interaction_history'])} electronics items")
        
        # Test different boost values
        boost_values = [1.0, 1.3, 1.6, 2.0]
        results = {}
        
        for boost in boost_values:
            print(f"\n📊 Testing boost = {boost}x:")
            
            try:
                recommendations = self.engine.recommend_items_raw_two_tower(
                    age=test_user['age'],
                    gender=test_user['gender'],
                    income=test_user['income'],
                    profession=test_user['profession'],
                    location=test_user['location'],
                    education_level=test_user['education_level'],
                    marital_status=test_user['marital_status'],
                    interaction_history=test_user['interaction_history'],
                    k=20,
                    exclude_history=True,
                    category_boost=boost
                )
                
                if not recommendations:
                    print(f"   ❌ No recommendations returned")
                    continue
                
                # Analyze category alignment
                electronics_count = 0
                construction_count = 0
                kids_count = 0  # User also has some kids items in history
                other_count = 0
                
                boosted_scores = []
                total_score = 0
                
                for item_id, score, item_info in recommendations:
                    category = item_info.get('category_code', '')
                    total_score += score
                    
                    # Check if this matches user's historical categories
                    if 'electronics' in category:
                        electronics_count += 1
                        boosted_scores.append(score)
                    elif 'construction' in category:
                        construction_count += 1
                        boosted_scores.append(score)
                    elif 'kids' in category:
                        kids_count += 1
                        boosted_scores.append(score)
                    else:
                        other_count += 1
                
                # Calculate metrics
                historical_categories_count = electronics_count + construction_count + kids_count
                category_alignment_pct = (historical_categories_count / len(recommendations)) * 100
                avg_score = total_score / len(recommendations)
                avg_boosted_score = np.mean(boosted_scores) if boosted_scores else 0
                
                results[boost] = {
                    'electronics_count': electronics_count,
                    'construction_count': construction_count,
                    'kids_count': kids_count,
                    'other_count': other_count,
                    'category_alignment_pct': category_alignment_pct,
                    'avg_score': avg_score,
                    'avg_boosted_score': avg_boosted_score,
                    'total_recs': len(recommendations)
                }
                
                print(f"   📈 Category Alignment: {historical_categories_count}/{len(recommendations)} ({category_alignment_pct:.1f}%)")
                print(f"   📊 Breakdown: Electronics({electronics_count}) Construction({construction_count}) Kids({kids_count}) Other({other_count})")
                print(f"   ⭐ Avg Score: {avg_score:.4f} | Avg Boosted Score: {avg_boosted_score:.4f}")
                
                # Show top 3 recommendations
                print(f"   🏆 Top 3:")
                for i, (item_id, score, item_info) in enumerate(recommendations[:3]):
                    category = item_info.get('category_code', 'Unknown')
                    brand = item_info.get('brand', 'Unknown')
                    is_historical = any(cat in category for cat in ['electronics', 'construction', 'kids'])
                    boost_indicator = "🔥" if is_historical else "🆕"
                    print(f"      {i+1}. {boost_indicator} {brand} - {category} (score: {score:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error with boost {boost}: {e}")
                continue
        
        # Compare results
        if len(results) >= 2:
            print(f"\n" + "="*70)
            print("📈 CATEGORY BOOST COMPARISON SUMMARY")
            print("="*70)
            
            print(f"{'Boost':<8} {'Alignment':<12} {'Avg Score':<12} {'Electronics':<12} {'Other':<8}")
            print("-" * 64)
            
            for boost in sorted(results.keys()):
                r = results[boost]
                print(f"{boost:<8} {r['category_alignment_pct']:<12.1f}% {r['avg_score']:<12.4f} {r['electronics_count']:<12} {r['other_count']:<8}")
            
            # Analysis
            print(f"\n🔍 Analysis:")
            
            # Compare 1.6x vs 1.3x specifically
            if 1.3 in results and 1.6 in results:
                r13 = results[1.3]
                r16 = results[1.6]
                
                alignment_diff = r16['category_alignment_pct'] - r13['category_alignment_pct']
                score_diff = r16['avg_score'] - r13['avg_score']
                electronics_diff = r16['electronics_count'] - r13['electronics_count']
                
                print(f"   📊 1.6x vs 1.3x boost comparison:")
                print(f"      Category Alignment: {alignment_diff:+.1f}% change")
                print(f"      Average Score: {score_diff:+.4f} change")
                print(f"      Electronics Items: {electronics_diff:+d} change")
                
                if alignment_diff > 5:
                    print(f"      ✅ Significant improvement in category alignment!")
                elif alignment_diff > 0:
                    print(f"      ⚠️  Slight improvement in category alignment")
                else:
                    print(f"      ❌ No improvement or decline in category alignment")
                
                # Score impact analysis
                if score_diff > 0:
                    print(f"      📈 Higher scores with 1.6x boost (better ranking)")
                else:
                    print(f"      📉 Lower scores with 1.6x boost (stronger boosting effect)")
        
        return results


def main():
    """Run the category boost comparison test."""
    
    try:
        tester = CategoryBoostTester()
        results = tester.test_category_boost_comparison()
        
        print(f"\n🎉 Category boost testing completed!")
        print(f"📝 Results show the impact of different category boost values")
        print(f"🎯 The new 1.6x default should provide stronger category preference alignment")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()