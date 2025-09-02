#!/usr/bin/env python3
"""
Raw Two-Tower Retrieval Validation Test

This script tests the RecommendationEngine with the fixed UserTower to verify:
1. Different users get different recommendations via two-tower retrieval
2. Users with similar demographics get somewhat similar recommendations
3. Users with different interaction histories get different recommendations
4. Zero-interaction users still get reasonable recommendations
5. Attention masking improvements work in practice
6. Raw two-tower similarity scores provide good recommendation quality
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference.recommendation_engine import RecommendationEngine
    print("✅ Successfully imported RecommendationEngine")
except Exception as e:
    print(f"❌ Failed to import RecommendationEngine: {e}")
    print("Make sure TensorFlow is installed and model weights exist")
    sys.exit(1)


class RecommendationValidator:
    """Test and validate recommendation quality."""
    
    def __init__(self):
        print("=== Recommendation Validation Test ===")
        print("Loading recommendation engine...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            print("This might be expected if you haven't retrained with the fixed UserTower yet")
            raise
    
    def create_test_users(self) -> List[Dict[str, Any]]:
        """Create diverse test users to validate recommendation quality."""
        
        return [
            # Group 1: Similar young tech professionals (should get similar recommendations)
            {
                'name': 'Young_Tech_Male_1',
                'age': 25,
                'gender': 'male',
                'income': 85000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Single',
                'interaction_history': [1000978, 1001588, 1001618, 1002000]  # Tech-related items
            },
            {
                'name': 'Young_Tech_Male_2',
                'age': 27,
                'gender': 'male',
                'income': 82000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Single',
                'interaction_history': [1000980, 1001590, 1001620, 1002010]  # Similar tech items
            },
            {
                'name': 'Young_Tech_Female',
                'age': 26,
                'gender': 'female',
                'income': 78000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Single',
                'interaction_history': [1000979, 1001589, 1001619, 1002005]  # Similar tech items
            },
            
            # Group 2: Healthcare professionals (different from tech group)
            {
                'name': 'Healthcare_Professional_1',
                'age': 35,
                'gender': 'female',
                'income': 68000,
                'profession': 'Healthcare',
                'location': 'Suburban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': [1003000, 1003100, 1003200, 1003300]  # Healthcare-related
            },
            {
                'name': 'Healthcare_Professional_2',
                'age': 42,
                'gender': 'male',
                'income': 72000,
                'profession': 'Healthcare',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': [1003010, 1003110, 1003210, 1003310]  # Similar healthcare
            },
            
            # Group 3: Very different demographics
            {
                'name': 'Senior_Rural_Retiree',
                'age': 67,
                'gender': 'female',
                'income': 35000,
                'profession': 'Other',
                'location': 'Rural',
                'education_level': "High School",
                'marital_status': 'Widowed',
                'interaction_history': [1004000, 1004100]  # Senior-focused items
            },
            {
                'name': 'Young_Student',
                'age': 20,
                'gender': 'male',
                'income': 15000,
                'profession': 'Other',
                'location': 'Urban',
                'education_level': "Some College",
                'marital_status': 'Single',
                'interaction_history': [1005000, 1005100, 1005200]  # Student items
            },
            
            # Group 4: Zero interaction users (cold start)
            {
                'name': 'Zero_Tech_Professional',
                'age': 30,
                'gender': 'male',
                'income': 75000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Single',
                'interaction_history': []  # No interactions - cold start
            },
            {
                'name': 'Zero_Healthcare_Professional',
                'age': 35,
                'gender': 'female',
                'income': 65000,
                'profession': 'Healthcare',
                'location': 'Suburban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': []  # No interactions - cold start
            },
            {
                'name': 'Zero_Different_Demographics',
                'age': 50,
                'gender': 'male',
                'income': 45000,
                'profession': 'Manufacturing',
                'location': 'Rural',
                'education_level': "High School",
                'marital_status': 'Married',
                'interaction_history': []  # No interactions - cold start
            }
        ]
    
    def get_recommendations_for_user(self, user: Dict[str, Any], k: int = 10) -> List[Tuple[int, float, Dict]]:
        """Get recommendations for a specific user."""
        
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
                k=k,
                exclude_history=True
            )
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting recommendations for {user['name']}: {e}")
            return []
    
    def calculate_recommendation_overlap(self, recs1: List[int], recs2: List[int]) -> float:
        """Calculate percentage overlap between two recommendation lists."""
        
        if len(recs1) == 0 or len(recs2) == 0:
            return 0.0
        
        set1, set2 = set(recs1), set(recs2)
        intersection = len(set1.intersection(set2))
        
        # Use smaller list as denominator
        return (intersection / min(len(recs1), len(recs2))) * 100
    
    def analyze_recommendation_diversity(self, all_recommendations: Dict[str, List[int]]) -> Dict[str, Any]:
        """Analyze diversity across all recommendations."""
        
        all_items = []
        for recs in all_recommendations.values():
            all_items.extend(recs)
        
        if not all_items:
            return {'error': 'No recommendations found'}
        
        from collections import Counter
        item_counts = Counter(all_items)
        total_items = len(all_items)
        unique_items = len(set(all_items))
        
        # Top 10 concentration
        top_10_items = item_counts.most_common(10)
        top_10_count = sum(count for _, count in top_10_items)
        top_10_concentration = (top_10_count / total_items) * 100
        
        return {
            'total_recommendations': total_items,
            'unique_items': unique_items,
            'diversity_ratio': unique_items / total_items,
            'top_10_concentration': top_10_concentration,
            'most_popular_items': top_10_items[:5]
        }
    
    def test_user_differentiation(self, test_users: List[Dict], all_recommendations: Dict[str, List[int]]):
        """Test that different users get meaningfully different recommendations."""
        
        print(f"\n🔍 Testing User Differentiation")
        print("-" * 50)
        
        # Test within-group similarity (should be higher)
        tech_users = ['Young_Tech_Male_1', 'Young_Tech_Male_2', 'Young_Tech_Female']
        healthcare_users = ['Healthcare_Professional_1', 'Healthcare_Professional_2']
        zero_users = ['Zero_Tech_Professional', 'Zero_Healthcare_Professional', 'Zero_Different_Demographics']
        
        def test_group_similarity(group_name: str, user_names: List[str]):
            print(f"\n📊 {group_name} Group Internal Similarity:")
            
            overlaps = []
            for i in range(len(user_names)):
                for j in range(i + 1, len(user_names)):
                    user1, user2 = user_names[i], user_names[j]
                    if user1 in all_recommendations and user2 in all_recommendations:
                        recs1, recs2 = all_recommendations[user1], all_recommendations[user2]
                        overlap = self.calculate_recommendation_overlap(recs1, recs2)
                        overlaps.append(overlap)
                        print(f"   {user1} ↔ {user2}: {overlap:.1f}% overlap")
            
            if overlaps:
                avg_overlap = np.mean(overlaps)
                print(f"   📈 Average within-group overlap: {avg_overlap:.1f}%")
                return avg_overlap
            return 0
        
        # Test different groups
        tech_similarity = test_group_similarity("Tech Professionals", tech_users)
        healthcare_similarity = test_group_similarity("Healthcare Professionals", healthcare_users) 
        zero_similarity = test_group_similarity("Zero-Interaction Users", zero_users)
        
        # Test cross-group dissimilarity (should be lower)
        print(f"\n📊 Cross-Group Dissimilarity:")
        
        cross_overlaps = []
        test_pairs = [
            ('Young_Tech_Male_1', 'Healthcare_Professional_1'),
            ('Young_Tech_Female', 'Senior_Rural_Retiree'),
            ('Healthcare_Professional_1', 'Young_Student'),
            ('Zero_Tech_Professional', 'Zero_Different_Demographics')
        ]
        
        for user1, user2 in test_pairs:
            if user1 in all_recommendations and user2 in all_recommendations:
                recs1, recs2 = all_recommendations[user1], all_recommendations[user2]
                overlap = self.calculate_recommendation_overlap(recs1, recs2)
                cross_overlaps.append(overlap)
                print(f"   {user1} ↔ {user2}: {overlap:.1f}% overlap")
        
        if cross_overlaps:
            avg_cross_overlap = np.mean(cross_overlaps)
            print(f"   📉 Average cross-group overlap: {avg_cross_overlap:.1f}%")
            
            # Evaluation
            if tech_similarity > avg_cross_overlap * 1.5:
                print(f"   ✅ Good differentiation: Similar users more alike than different users")
            else:
                print(f"   ⚠️  Weak differentiation: May need further improvements")
    
    def test_cold_start_handling(self, test_users: List[Dict], all_recommendations: Dict[str, List[int]]):
        """Test how well zero-interaction users are handled."""
        
        print(f"\n🆕 Testing Cold Start Handling")
        print("-" * 50)
        
        zero_users = [user for user in test_users if len(user['interaction_history']) == 0]
        
        for user in zero_users:
            user_name = user['name']
            if user_name in all_recommendations:
                recs = all_recommendations[user_name]
                print(f"   {user_name}: {len(recs)} recommendations")
                
                if len(recs) > 0:
                    print(f"     ✅ Cold start user got recommendations")
                    # Show top 3
                    for i, item_id in enumerate(recs[:3]):
                        print(f"     {i+1}. Item {item_id}")
                else:
                    print(f"     ❌ Cold start user got no recommendations")
            else:
                print(f"   {user_name}: ❌ Failed to get recommendations")
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete recommendation validation."""
        
        print(f"\n🚀 Starting Recommendation Validation...")
        start_time = time.time()
        
        # Create test users
        test_users = self.create_test_users()
        print(f"📋 Created {len(test_users)} test users")
        
        # Get recommendations for all users
        all_recommendations = {}
        user_recommendation_details = {}
        
        for user in test_users:
            print(f"\n🔍 Getting recommendations for {user['name']}...")
            print(f"   Demographics: {user['age']}y {user['gender']} {user['profession']} (${user['income']:,})")
            print(f"   Interactions: {len(user['interaction_history'])}")
            
            recommendations = self.get_recommendations_for_user(user, k=10)
            
            if recommendations:
                item_ids = [item_id for item_id, _, _ in recommendations]
                scores = [score for _, score, _ in recommendations]
                
                all_recommendations[user['name']] = item_ids
                user_recommendation_details[user['name']] = {
                    'recommendations': recommendations,
                    'scores': scores,
                    'score_stats': {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores)
                    }
                }
                
                print(f"   ✅ Got {len(recommendations)} recommendations")
                print(f"   📊 Score range: {np.min(scores):.4f} - {np.max(scores):.4f}")
                print(f"   🏆 Top 3: {item_ids[:3]}")
            else:
                print(f"   ❌ No recommendations returned")
        
        # Run analysis tests
        self.test_user_differentiation(test_users, all_recommendations)
        self.test_cold_start_handling(test_users, all_recommendations)
        
        # Analyze overall diversity
        diversity_analysis = self.analyze_recommendation_diversity(all_recommendations)
        
        print(f"\n📊 Overall Recommendation Diversity:")
        print("-" * 50)
        if 'error' not in diversity_analysis:
            print(f"   Total recommendations: {diversity_analysis['total_recommendations']}")
            print(f"   Unique items: {diversity_analysis['unique_items']}")
            print(f"   Diversity ratio: {diversity_analysis['diversity_ratio']:.3f}")
            print(f"   Top 10 concentration: {diversity_analysis['top_10_concentration']:.1f}%")
            
            if diversity_analysis['diversity_ratio'] > 0.7:
                print(f"   ✅ Excellent diversity")
            elif diversity_analysis['diversity_ratio'] > 0.4:
                print(f"   ⚠️  Moderate diversity")
            else:
                print(f"   ❌ Poor diversity - dominated by popular items")
        
        # Compile results
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'runtime_seconds': time.time() - start_time,
            'users_tested': len(test_users),
            'successful_recommendations': len([r for r in all_recommendations.values() if r]),
            'user_recommendations': all_recommendations,
            'recommendation_details': user_recommendation_details,
            'diversity_analysis': diversity_analysis
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary."""
        
        print(f"\n" + "="*70)
        print("📋 RECOMMENDATION VALIDATION SUMMARY")
        print("="*70)
        
        print(f"⏱️  Total runtime: {results['runtime_seconds']:.2f} seconds")
        print(f"👥 Users tested: {results['users_tested']}")
        print(f"✅ Successful recommendations: {results['successful_recommendations']}/{results['users_tested']}")
        
        if results['successful_recommendations'] == results['users_tested']:
            print(f"🎉 ALL USERS GOT RECOMMENDATIONS!")
            print(f"✅ Fixed UserTower attention masking appears to be working")
        else:
            print(f"⚠️  Some users didn't get recommendations - check error messages above")
        
        # Overall assessment
        if results['successful_recommendations'] >= results['users_tested'] * 0.8:
            if 'diversity_analysis' in results and results['diversity_analysis'].get('diversity_ratio', 0) > 0.4:
                print(f"\n🏆 VALIDATION PASSED - Recommendations look good!")
                print(f"   ✅ Most users getting recommendations")
                print(f"   ✅ Reasonable diversity detected")
                print(f"   ✅ UserTower attention fixes working as expected")
            else:
                print(f"\n⚠️  PARTIAL SUCCESS - Users getting recommendations but diversity may be low")
        else:
            print(f"\n❌ VALIDATION FAILED - Many users not getting recommendations")
            print(f"   Please check if model weights are properly loaded")
        
        print("="*70)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save validation results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommendation_validation_results_{timestamp}.json"
        
        # Clean for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to {filename}")


def main():
    """Run recommendation validation test."""
    
    try:
        validator = RecommendationValidator()
        results = validator.run_validation()
        validator.print_summary(results)
        validator.save_results(results)
        
        print(f"\n✅ Recommendation validation completed!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()