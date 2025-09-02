#!/usr/bin/env python3
"""
Raw Two-Tower Retrieval Overlap Analysis

This test analyzes raw two-tower retrieval recommendations to check:
1. Item overlap between similar users (should have high overlap)
2. Item overlap between dissimilar users (should have low overlap)
3. Overall two-tower similarity signal strength
4. Demographic vs interaction history influence
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set
import json
from collections import defaultdict, Counter

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.recommendation_engine import RecommendationEngine


class CollaborativeOverlapTester:
    """Test collaborative filtering overlap patterns between users."""
    
    def __init__(self):
        print("=== Collaborative Filtering Overlap Analysis ===")
        print("Loading recommendation engine...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            raise
        
        self.test_results = []
        
    def create_user_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create groups of similar and dissimilar users for overlap analysis."""
        
        print("\n📋 Creating user groups for overlap analysis...")
        
        user_groups = {
            # Similar users - Young Tech Professionals
            'young_tech_similar': [
                {
                    'name': 'Young_Tech_Male_1',
                    'age': 25,
                    'gender': 'male',
                    'income': 85000,
                    'profession': 'Technology',
                    'location': 'Urban',
                    'education_level': "Bachelor's",
                    'marital_status': 'Single',
                    'interaction_history': [1000978, 1001588, 1001618]  # Tech items
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
                    'interaction_history': [1000980, 1001590, 1001620]  # Similar tech items
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
                    'interaction_history': [1000979, 1001589, 1001619]  # Similar tech items
                }
            ],
            
            # Similar users - Healthcare Workers
            'healthcare_similar': [
                {
                    'name': 'Healthcare_Female_1',
                    'age': 35,
                    'gender': 'female',
                    'income': 68000,
                    'profession': 'Healthcare',
                    'location': 'Suburban',
                    'education_level': "Master's",
                    'marital_status': 'Married',
                    'interaction_history': [1002000, 1002100, 1002200]  # Healthcare-relevant items
                },
                {
                    'name': 'Healthcare_Male',
                    'age': 42,
                    'gender': 'male',
                    'income': 72000,
                    'profession': 'Healthcare',
                    'location': 'Urban',
                    'education_level': "Master's",
                    'marital_status': 'Married',
                    'interaction_history': [1002010, 1002110, 1002210]  # Similar healthcare items
                },
                {
                    'name': 'Healthcare_Female_2',
                    'age': 38,
                    'gender': 'female',
                    'income': 65000,
                    'profession': 'Healthcare',
                    'location': 'Suburban',
                    'education_level': "Bachelor's",
                    'marital_status': 'Married',
                    'interaction_history': [1002020, 1002120, 1002220]  # Similar healthcare items
                }
            ],
            
            # Dissimilar users - Mixed demographics
            'mixed_dissimilar': [
                {
                    'name': 'Young_Student',
                    'age': 20,
                    'gender': 'male',
                    'income': 15000,
                    'profession': 'Other',
                    'location': 'Urban',
                    'education_level': "Some College",
                    'marital_status': 'Single',
                    'interaction_history': [1003000, 1003100]  # Student items
                },
                {
                    'name': 'Senior_Retiree',
                    'age': 67,
                    'gender': 'female',
                    'income': 45000,
                    'profession': 'Other',
                    'location': 'Rural',
                    'education_level': "High School",
                    'marital_status': 'Widowed',
                    'interaction_history': [1004000, 1004100]  # Senior items
                },
                {
                    'name': 'Mid_Manufacturing',
                    'age': 43,
                    'gender': 'male',
                    'income': 54000,
                    'profession': 'Manufacturing',
                    'location': 'Rural',
                    'education_level': "High School",
                    'marital_status': 'Married',
                    'interaction_history': [1005000, 1005100]  # Manufacturing items
                }
            ],
            
            # Zero interaction users for baseline
            'zero_interaction': [
                {
                    'name': 'Zero_Tech',
                    'age': 28,
                    'gender': 'male',
                    'income': 75000,
                    'profession': 'Technology',
                    'location': 'Urban',
                    'education_level': "Bachelor's",
                    'marital_status': 'Single',
                    'interaction_history': []  # No interactions
                },
                {
                    'name': 'Zero_Healthcare',
                    'age': 35,
                    'gender': 'female',
                    'income': 65000,
                    'profession': 'Healthcare',
                    'location': 'Suburban',
                    'education_level': "Master's",
                    'marital_status': 'Married',
                    'interaction_history': []  # No interactions
                },
                {
                    'name': 'Zero_Other',
                    'age': 45,
                    'gender': 'male',
                    'income': 50000,
                    'profession': 'Other',
                    'location': 'Rural',
                    'education_level': "High School",
                    'marital_status': 'Married',
                    'interaction_history': []  # No interactions
                }
            ]
        }
        
        return user_groups
    
    def get_collaborative_recommendations(self, user_profile: Dict[str, Any], k: int = 20) -> List[int]:
        """Get collaborative filtering recommendations for a user."""
        
        try:
            recommendations = self.engine.recommend_items_raw_two_tower(
                age=user_profile['age'],
                gender=user_profile['gender'],
                income=user_profile['income'],
                profession=user_profile['profession'],
                location=user_profile['location'],
                education_level=user_profile['education_level'],
                marital_status=user_profile['marital_status'],
                interaction_history=user_profile['interaction_history'],
                k=k,
                exclude_history=True
            )
            
            # Extract just the item IDs
            return [item_id for item_id, _, _ in recommendations]
            
        except Exception as e:
            print(f"   ❌ Error getting recommendations for {user_profile['name']}: {e}")
            return []
    
    def calculate_jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """Calculate Jaccard similarity between two sets of items."""
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0  # Both empty sets are identical
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_overlap_percentage(self, list1: List[int], list2: List[int]) -> float:
        """Calculate percentage of items that overlap between two recommendation lists."""
        
        if len(list1) == 0 or len(list2) == 0:
            return 0.0
        
        set1, set2 = set(list1), set(list2)
        intersection = len(set1.intersection(set2))
        
        # Use the smaller list as denominator for percentage
        return (intersection / min(len(list1), len(list2))) * 100
    
    def analyze_group_overlap(self, group_name: str, users: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overlap patterns within a group of users."""
        
        print(f"\n🔍 Analyzing overlap for group: {group_name}")
        
        # Get recommendations for all users in the group
        user_recommendations = {}
        
        for user in users:
            print(f"   Getting recommendations for {user['name']}...")
            recs = self.get_collaborative_recommendations(user)
            user_recommendations[user['name']] = recs
            
            if len(recs) > 0:
                print(f"     ✅ Got {len(recs)} recommendations")
            else:
                print(f"     ❌ No recommendations returned")
        
        # Calculate pairwise overlaps
        user_names = list(user_recommendations.keys())
        pairwise_overlaps = []
        pairwise_jaccard = []
        
        for i in range(len(user_names)):
            for j in range(i + 1, len(user_names)):
                user1, user2 = user_names[i], user_names[j]
                recs1, recs2 = user_recommendations[user1], user_recommendations[user2]
                
                if len(recs1) > 0 and len(recs2) > 0:
                    overlap_pct = self.calculate_overlap_percentage(recs1, recs2)
                    jaccard = self.calculate_jaccard_similarity(set(recs1), set(recs2))
                    
                    pairwise_overlaps.append(overlap_pct)
                    pairwise_jaccard.append(jaccard)
                    
                    print(f"     {user1} ↔ {user2}: {overlap_pct:.1f}% overlap, Jaccard: {jaccard:.3f}")
        
        # Calculate group statistics
        if pairwise_overlaps:
            group_stats = {
                'group_name': group_name,
                'user_count': len(users),
                'successful_recommendations': sum(1 for recs in user_recommendations.values() if len(recs) > 0),
                'mean_overlap_percentage': np.mean(pairwise_overlaps),
                'std_overlap_percentage': np.std(pairwise_overlaps),
                'min_overlap_percentage': np.min(pairwise_overlaps),
                'max_overlap_percentage': np.max(pairwise_overlaps),
                'mean_jaccard_similarity': np.mean(pairwise_jaccard),
                'std_jaccard_similarity': np.std(pairwise_jaccard),
                'user_recommendations': user_recommendations,
                'pairwise_overlaps': pairwise_overlaps,
                'pairwise_jaccard': pairwise_jaccard
            }
        else:
            group_stats = {
                'group_name': group_name,
                'user_count': len(users),
                'successful_recommendations': 0,
                'error': 'No valid recommendation pairs found'
            }
        
        return group_stats
    
    def analyze_cross_group_overlap(self, group1_stats: Dict, group2_stats: Dict) -> Dict[str, Any]:
        """Analyze overlap between users from different groups."""
        
        print(f"\n🔀 Analyzing cross-group overlap: {group1_stats['group_name']} vs {group2_stats['group_name']}")
        
        cross_overlaps = []
        cross_jaccard = []
        
        recs1 = group1_stats.get('user_recommendations', {})
        recs2 = group2_stats.get('user_recommendations', {})
        
        for user1, user1_recs in recs1.items():
            for user2, user2_recs in recs2.items():
                if len(user1_recs) > 0 and len(user2_recs) > 0:
                    overlap_pct = self.calculate_overlap_percentage(user1_recs, user2_recs)
                    jaccard = self.calculate_jaccard_similarity(set(user1_recs), set(user2_recs))
                    
                    cross_overlaps.append(overlap_pct)
                    cross_jaccard.append(jaccard)
                    
                    print(f"   {user1} ↔ {user2}: {overlap_pct:.1f}% overlap, Jaccard: {jaccard:.3f}")
        
        if cross_overlaps:
            return {
                'group1': group1_stats['group_name'],
                'group2': group2_stats['group_name'],
                'mean_overlap_percentage': np.mean(cross_overlaps),
                'std_overlap_percentage': np.std(cross_overlaps),
                'mean_jaccard_similarity': np.mean(cross_jaccard),
                'comparison_count': len(cross_overlaps)
            }
        else:
            return {
                'group1': group1_stats['group_name'],
                'group2': group2_stats['group_name'],
                'error': 'No valid cross-group comparisons found'
            }
    
    def analyze_popular_items(self, all_group_stats: List[Dict]) -> Dict[str, Any]:
        """Analyze which items appear most frequently across all recommendations."""
        
        print(f"\n📊 Analyzing popular items across all groups...")
        
        all_recommendations = []
        
        for group_stats in all_group_stats:
            if 'user_recommendations' in group_stats:
                for user, recs in group_stats['user_recommendations'].items():
                    all_recommendations.extend(recs)
        
        if not all_recommendations:
            return {'error': 'No recommendations found for popularity analysis'}
        
        # Count item frequencies
        item_counts = Counter(all_recommendations)
        total_items = len(all_recommendations)
        unique_items = len(set(all_recommendations))
        
        # Get top items
        top_items = item_counts.most_common(20)
        
        # Calculate concentration metrics
        top_10_percentage = sum(count for _, count in top_items[:10]) / total_items * 100
        
        return {
            'total_recommendations': total_items,
            'unique_items': unique_items,
            'diversity_ratio': unique_items / total_items,
            'top_10_concentration': top_10_percentage,
            'top_items': top_items,
            'most_popular_item': top_items[0] if top_items else None
        }
    
    def run_overlap_analysis(self) -> Dict[str, Any]:
        """Run the complete collaborative filtering overlap analysis."""
        
        print(f"\n🚀 Starting collaborative filtering overlap analysis...")
        start_time = time.time()
        
        # Create user groups
        user_groups = self.create_user_groups()
        
        # Analyze each group
        group_results = {}
        
        for group_name, users in user_groups.items():
            group_results[group_name] = self.analyze_group_overlap(group_name, users)
        
        # Analyze cross-group overlaps
        cross_group_results = []
        
        # Similar groups should have higher overlap than dissimilar groups
        group_pairs = [
            ('young_tech_similar', 'healthcare_similar'),  # Different professions
            ('young_tech_similar', 'mixed_dissimilar'),    # Very different
            ('healthcare_similar', 'mixed_dissimilar'),    # Very different
            ('young_tech_similar', 'zero_interaction'),    # With vs without history
            ('healthcare_similar', 'zero_interaction'),    # With vs without history
        ]
        
        for group1, group2 in group_pairs:
            if group1 in group_results and group2 in group_results:
                cross_result = self.analyze_cross_group_overlap(
                    group_results[group1], 
                    group_results[group2]
                )
                cross_group_results.append(cross_result)
        
        # Analyze popular items
        popular_items_analysis = self.analyze_popular_items(list(group_results.values()))
        
        # Compile final results
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_runtime_seconds': time.time() - start_time,
            'group_analyses': group_results,
            'cross_group_analyses': cross_group_results,
            'popular_items_analysis': popular_items_analysis
        }
        
        return results
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a summary report of the overlap analysis."""
        
        print(f"\n" + "="*60)
        print("📋 COLLABORATIVE FILTERING OVERLAP ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"⏱️  Analysis completed in {results['total_runtime_seconds']:.2f} seconds")
        
        # Group analysis summary
        print(f"\n📊 Within-Group Overlap Results:")
        print("-" * 40)
        
        for group_name, stats in results['group_analyses'].items():
            if 'mean_overlap_percentage' in stats:
                print(f"{group_name:25}: {stats['mean_overlap_percentage']:.1f}% ± {stats['std_overlap_percentage']:.1f}% "
                      f"(Jaccard: {stats['mean_jaccard_similarity']:.3f})")
            else:
                print(f"{group_name:25}: ❌ No valid results")
        
        # Cross-group analysis summary
        print(f"\n🔀 Cross-Group Overlap Results:")
        print("-" * 40)
        
        for cross_stats in results['cross_group_analyses']:
            if 'mean_overlap_percentage' in cross_stats:
                print(f"{cross_stats['group1']} ↔ {cross_stats['group2']}: "
                      f"{cross_stats['mean_overlap_percentage']:.1f}% "
                      f"(Jaccard: {cross_stats['mean_jaccard_similarity']:.3f})")
            else:
                print(f"{cross_stats['group1']} ↔ {cross_stats['group2']}: ❌ No valid results")
        
        # Popular items analysis
        pop_analysis = results['popular_items_analysis']
        if 'error' not in pop_analysis:
            print(f"\n🏆 Popular Items Analysis:")
            print("-" * 40)
            print(f"Total recommendations: {pop_analysis['total_recommendations']}")
            print(f"Unique items: {pop_analysis['unique_items']}")
            print(f"Diversity ratio: {pop_analysis['diversity_ratio']:.3f}")
            print(f"Top 10 concentration: {pop_analysis['top_10_concentration']:.1f}%")
            
            if pop_analysis['most_popular_item']:
                item_id, count = pop_analysis['most_popular_item']
                percentage = (count / pop_analysis['total_recommendations']) * 100
                print(f"Most popular item: {item_id} ({count} times, {percentage:.1f}%)")
        
        # Interpretation
        print(f"\n🔍 Interpretation:")
        print("-" * 40)
        
        # Check if collaborative filtering is working
        similar_overlaps = []
        dissimilar_overlaps = []
        
        for group_name, stats in results['group_analyses'].items():
            if 'mean_overlap_percentage' in stats:
                if 'similar' in group_name:
                    similar_overlaps.append(stats['mean_overlap_percentage'])
                elif 'dissimilar' in group_name:
                    dissimilar_overlaps.append(stats['mean_overlap_percentage'])
        
        if similar_overlaps and dissimilar_overlaps:
            avg_similar = np.mean(similar_overlaps)
            avg_dissimilar = np.mean(dissimilar_overlaps)
            
            if avg_similar > avg_dissimilar * 1.5:
                print("✅ Strong collaborative filtering signal detected")
                print(f"   Similar users have {avg_similar:.1f}% overlap vs {avg_dissimilar:.1f}% for dissimilar users")
            elif avg_similar > avg_dissimilar:
                print("⚠️  Weak collaborative filtering signal detected")
                print(f"   Similar users have {avg_similar:.1f}% overlap vs {avg_dissimilar:.1f}% for dissimilar users")
            else:
                print("❌ No clear collaborative filtering signal")
                print("   Recommendations may be dominated by popularity or demographic bias")
        
        # Check diversity
        if 'error' not in pop_analysis:
            if pop_analysis['diversity_ratio'] > 0.7:
                print("✅ Good recommendation diversity")
            elif pop_analysis['diversity_ratio'] > 0.4:
                print("⚠️  Moderate recommendation diversity")
            else:
                print("❌ Poor recommendation diversity - dominated by popular items")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collaborative_overlap_analysis_{timestamp}.json"
        
        # Clean results for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to {filename}")


def main():
    """Main function to run the collaborative filtering overlap analysis."""
    
    try:
        tester = CollaborativeOverlapTester()
        results = tester.run_overlap_analysis()
        tester.print_summary_report(results)
        tester.save_results(results)
        
        print(f"\n✅ Collaborative filtering overlap analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()