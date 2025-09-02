#!/usr/bin/env python3
"""
Zero-Interaction User Recommendations Test

This script comprehensively tests recommendation algorithms for users with no interaction history
to evaluate the effectiveness of demographic boosting vs other cold-start approaches.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.recommendation_engine import RecommendationEngine


class ZeroInteractionTester:
    """Test recommendation algorithms specifically for users with zero interactions."""
    
    def __init__(self):
        print("=== Zero-Interaction User Recommendation Tester ===")
        print("Loading recommendation engine...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            raise
        
        self.test_results = []
        self.algorithms = ['collaborative', 'demographic_boosted', 'hybrid']
        
    def create_test_profiles(self) -> List[Dict[str, Any]]:
        """Create diverse demographic profiles for testing."""
        
        print("\n📋 Creating diverse zero-interaction test profiles...")
        
        profiles = [
            # Young Tech Professionals
            {
                'name': 'Young_Tech_Male_Urban_High',
                'profile_type': 'young_tech',
                'age': 25,
                'gender': 'male',
                'income': 85000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Single',
                'interaction_history': []  # ZERO interactions
            },
            {
                'name': 'Young_Tech_Female_Urban_High',
                'profile_type': 'young_tech',
                'age': 27,
                'gender': 'female',
                'income': 78000,
                'profession': 'Technology',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Single',
                'interaction_history': []
            },
            
            # Healthcare Workers
            {
                'name': 'Mid_Healthcare_Female_Sub_Med',
                'profile_type': 'healthcare_mid',
                'age': 38,
                'gender': 'female',
                'income': 65000,
                'profession': 'Healthcare',
                'location': 'Suburban',
                'education_level': "Bachelor's",
                'marital_status': 'Married',
                'interaction_history': []
            },
            {
                'name': 'Mid_Healthcare_Male_Urban_Med',
                'profile_type': 'healthcare_mid',
                'age': 42,
                'gender': 'male',
                'income': 72000,
                'profession': 'Healthcare',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': []
            },
            
            # Finance Professionals
            {
                'name': 'Senior_Finance_Male_Urban_VHigh',
                'profile_type': 'finance_senior',
                'age': 55,
                'gender': 'male',
                'income': 135000,
                'profession': 'Finance',
                'location': 'Urban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': []
            },
            {
                'name': 'Mid_Finance_Female_Urban_High',
                'profile_type': 'finance_mid',
                'age': 45,
                'gender': 'female',
                'income': 95000,
                'profession': 'Finance',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Divorced',
                'interaction_history': []
            },
            
            # Retail Workers (Lower Income)
            {
                'name': 'Young_Retail_Female_Sub_Low',
                'profile_type': 'retail_young',
                'age': 24,
                'gender': 'female',
                'income': 32000,
                'profession': 'Retail',
                'location': 'Suburban',
                'education_level': 'High School',
                'marital_status': 'Single',
                'interaction_history': []
            },
            {
                'name': 'Mid_Retail_Male_Rural_Low',
                'profile_type': 'retail_mid',
                'age': 35,
                'gender': 'male',
                'income': 38000,
                'profession': 'Retail',
                'location': 'Rural',
                'education_level': 'Some College',
                'marital_status': 'Married',
                'interaction_history': []
            },
            
            # Education Professionals
            {
                'name': 'Mid_Education_Female_Sub_Med',
                'profile_type': 'education_mid',
                'age': 41,
                'gender': 'female',
                'income': 58000,
                'profession': 'Education',
                'location': 'Suburban',
                'education_level': "Master's",
                'marital_status': 'Married',
                'interaction_history': []
            },
            {
                'name': 'Senior_Education_Male_Rural_Med',
                'profile_type': 'education_senior',
                'age': 58,
                'gender': 'male',
                'income': 62000,
                'profession': 'Education',
                'location': 'Rural',
                'education_level': 'PhD+',
                'marital_status': 'Married',
                'interaction_history': []
            },
            
            # Manufacturing Workers
            {
                'name': 'Mid_Manufacturing_Male_Rural_Med',
                'profile_type': 'manufacturing_mid',
                'age': 43,
                'gender': 'male',
                'income': 54000,
                'profession': 'Manufacturing',
                'location': 'Rural',
                'education_level': 'High School',
                'marital_status': 'Married',
                'interaction_history': []
            },
            {
                'name': 'Young_Manufacturing_Female_Sub_Med',
                'profile_type': 'manufacturing_young',
                'age': 29,
                'gender': 'female',
                'income': 48000,
                'profession': 'Manufacturing',
                'location': 'Suburban',
                'education_level': 'Some College',
                'marital_status': 'Single',
                'interaction_history': []
            },
            
            # Services Workers
            {
                'name': 'Mid_Services_Female_Urban_Med',
                'profile_type': 'services_mid',
                'age': 37,
                'gender': 'female',
                'income': 51000,
                'profession': 'Services',
                'location': 'Urban',
                'education_level': "Bachelor's",
                'marital_status': 'Divorced',
                'interaction_history': []
            },
            {
                'name': 'Senior_Services_Male_Sub_Med',
                'profile_type': 'services_senior',
                'age': 61,
                'gender': 'male',
                'income': 56000,
                'profession': 'Services',
                'location': 'Suburban',
                'education_level': 'High School',
                'marital_status': 'Widowed',
                'interaction_history': []
            },
            
            # Edge Cases
            {
                'name': 'Very_Young_Other_Rural_Low',
                'profile_type': 'edge_young',
                'age': 18,
                'gender': 'male',
                'income': 28000,
                'profession': 'Other',
                'location': 'Rural',
                'education_level': 'High School',
                'marital_status': 'Single',
                'interaction_history': []
            },
            {
                'name': 'Senior_Other_Urban_High',
                'profile_type': 'edge_senior',
                'age': 67,
                'gender': 'female',
                'income': 110000,
                'profession': 'Other',
                'location': 'Urban',
                'education_level': 'PhD+',
                'marital_status': 'Widowed',
                'interaction_history': []
            }
        ]
        
        print(f"✅ Created {len(profiles)} diverse test profiles")
        
        # Verify all profiles have zero interactions
        for profile in profiles:
            assert len(profile['interaction_history']) == 0, f"Profile {profile['name']} has interactions!"
        
        return profiles
    
    def test_recommendation_algorithm(self, profile: Dict[str, Any], algorithm: str, k: int = 10) -> Dict[str, Any]:
        """Test a specific recommendation algorithm for a zero-interaction user."""
        
        result = {
            'success': False,
            'algorithm': algorithm,
            'profile_name': profile['name'],
            'recommendations': [],
            'recommendation_count': 0,
            'mean_score': 0.0,
            'std_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'category_diversity': 0,
            'top_categories': {},
            'elapsed_time': 0.0,
            'error_message': None
        }
        
        try:
            start_time = time.time()
            
            # Call the appropriate recommendation method
            if algorithm == 'collaborative':
                recommendations = self.engine.recommend_items_collaborative(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'],
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],  # Empty list
                    k=k
                )
                
            elif algorithm == 'demographic_boosted':
                recommendations = self.engine.recommend_items_demographic_boosted(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'],
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],  # Empty list
                    k=k
                )
                
            elif algorithm == 'hybrid':
                recommendations = self.engine.recommend_items_hybrid(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'],
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],  # Empty list
                    k=k
                )
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            elapsed_time = time.time() - start_time
            
            # Process results
            if recommendations:
                result['success'] = True
                result['recommendations'] = recommendations
                result['recommendation_count'] = len(recommendations)
                result['elapsed_time'] = elapsed_time
                
                # Calculate score statistics
                scores = [score for _, score, _ in recommendations]
                if scores:
                    result['mean_score'] = np.mean(scores)
                    result['std_score'] = np.std(scores)
                    result['min_score'] = np.min(scores)
                    result['max_score'] = np.max(scores)
                
                # Calculate category diversity
                categories = [info.get('category_code', '').split('.')[0] for _, _, info in recommendations]
                unique_categories = set(cat for cat in categories if cat)
                result['category_diversity'] = len(unique_categories)
                
                # Count top categories
                from collections import Counter
                category_counts = Counter(categories)
                result['top_categories'] = dict(category_counts.most_common(5))
                
            else:
                result['error_message'] = "No recommendations returned"
                
        except Exception as e:
            result['error_message'] = str(e)
            result['elapsed_time'] = time.time() - start_time
            
        return result
    
    def run_comprehensive_test(self, k: int = 10) -> None:
        """Run comprehensive tests across all profiles and algorithms."""
        
        print(f"\n🚀 Starting comprehensive zero-interaction test (k={k})...")
        
        # Get test profiles
        profiles = self.create_test_profiles()
        
        # Test each profile with each algorithm
        total_tests = len(profiles) * len(self.algorithms)
        current_test = 0
        
        print(f"\n📊 Running {total_tests} tests ({len(profiles)} profiles × {len(self.algorithms)} algorithms)...")
        
        for profile in profiles:
            print(f"\n👤 Testing profile: {profile['name']}")
            print(f"   Demographics: {profile['age']}y {profile['gender']} ${profile['income']} {profile['profession']}")
            print(f"   Location: {profile['location']}, Education: {profile['education_level']}, Marital: {profile['marital_status']}")
            print(f"   Interaction History: {len(profile['interaction_history'])} items (ZERO - cold start)")
            
            profile_results = {
                'profile': profile,
                'algorithm_results': {}
            }
            
            # Test each algorithm
            for algorithm in self.algorithms:
                current_test += 1
                print(f"      [{current_test}/{total_tests}] Testing {algorithm}...")
                
                result = self.test_recommendation_algorithm(profile, algorithm, k)
                profile_results['algorithm_results'][algorithm] = result
                
                if result['success']:
                    print(f"         ✅ {result['recommendation_count']} recs, "
                          f"score: {result['mean_score']:.4f}±{result['std_score']:.4f}, "
                          f"diversity: {result['category_diversity']}, "
                          f"time: {result['elapsed_time']:.3f}s")
                else:
                    print(f"         ❌ Failed: {result['error_message']}")
            
            self.test_results.append(profile_results)
        
        print(f"\n✅ Comprehensive testing completed!")
    
    def analyze_results(self) -> None:
        """Analyze and summarize test results."""
        
        print(f"\n📈 ZERO-INTERACTION RECOMMENDATION ANALYSIS")
        print(f"=" * 60)
        
        if not self.test_results:
            print("❌ No test results available")
            return
        
        print(f"   Total Profiles Tested: {len(self.test_results)}")
        print(f"   Algorithms Compared: {', '.join(self.algorithms)}")
        
        # Algorithm performance comparison
        print(f"\n🎯 ALGORITHM PERFORMANCE COMPARISON:")
        
        algorithm_stats = {}
        
        for algorithm in self.algorithms:
            successes = 0
            scores = []
            diversities = []
            times = []
            
            for result in self.test_results:
                algo_result = result['algorithm_results'][algorithm]
                if algo_result['success']:
                    successes += 1
                    scores.append(algo_result['mean_score'])
                    diversities.append(algo_result['category_diversity'])
                    times.append(algo_result['elapsed_time'])
            
            success_rate = successes / len(self.test_results) * 100
            
            algorithm_stats[algorithm] = {
                'success_rate': success_rate,
                'successful_tests': successes,
                'total_tests': len(self.test_results),
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'mean_diversity': np.mean(diversities) if diversities else 0,
                'mean_time': np.mean(times) if times else 0,
                'scores': scores,
                'diversities': diversities
            }
            
            print(f"\n   === {algorithm.upper()} ===")
            print(f"   Success Rate: {success_rate:.1f}% ({successes}/{len(self.test_results)})")
            
            if scores:
                print(f"   Mean Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
                print(f"   Score Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
                print(f"   Mean Diversity: {np.mean(diversities):.1f} categories")
                print(f"   Mean Response Time: {np.mean(times)*1000:.1f}ms")
            
        # Best algorithm analysis
        print(f"\n🏆 BEST ALGORITHM FOR ZERO-INTERACTION USERS:")
        
        best_success = max(algorithm_stats.values(), key=lambda x: x['success_rate'])
        best_score = max((algo for algo in algorithm_stats.values() if algo['scores']), 
                        key=lambda x: x['mean_score'], default=None)
        best_diversity = max((algo for algo in algorithm_stats.values() if algo['diversities']), 
                           key=lambda x: x['mean_diversity'], default=None)
        
        for algo, stats in algorithm_stats.items():
            if stats == best_success:
                print(f"   🎯 Highest Success Rate: {algo} ({stats['success_rate']:.1f}%)")
            if best_score and stats == best_score:
                print(f"   📊 Highest Scores: {algo} ({stats['mean_score']:.4f})")
            if best_diversity and stats == best_diversity:
                print(f"   🌈 Most Diverse: {algo} ({stats['mean_diversity']:.1f} categories)")
        
        # Profile type analysis
        print(f"\n👥 PROFILE TYPE ANALYSIS:")
        
        profile_type_results = {}
        for result in self.test_results:
            profile_type = result['profile']['profile_type']
            if profile_type not in profile_type_results:
                profile_type_results[profile_type] = []
            profile_type_results[profile_type].append(result)
        
        for profile_type, results in profile_type_results.items():
            print(f"\n   {profile_type.replace('_', ' ').title()} ({len(results)} profiles):")
            
            for algorithm in self.algorithms:
                successes = sum(1 for r in results if r['algorithm_results'][algorithm]['success'])
                if successes > 0:
                    scores = [r['algorithm_results'][algorithm]['mean_score'] 
                             for r in results if r['algorithm_results'][algorithm]['success']]
                    print(f"      {algorithm}: {successes}/{len(results)} success, "
                          f"avg score: {np.mean(scores):.4f}")
        
        # Summary insights
        print(f"\n💡 KEY INSIGHTS:")
        
        if 'demographic_boosted' in algorithm_stats and 'collaborative' in algorithm_stats:
            demo_stats = algorithm_stats['demographic_boosted']
            collab_stats = algorithm_stats['collaborative']
            
            if demo_stats['mean_score'] > collab_stats['mean_score']:
                improvement = ((demo_stats['mean_score'] - collab_stats['mean_score']) / collab_stats['mean_score']) * 100
                print(f"   📈 Demographic boosting improves scores by {improvement:.1f}% vs collaborative filtering")
            
            if demo_stats['mean_diversity'] < collab_stats['mean_diversity']:
                reduction = collab_stats['mean_diversity'] - demo_stats['mean_diversity']
                print(f"   🎯 Demographic boosting reduces diversity by {reduction:.1f} categories (more focused)")
        
        print(f"   🔥 Zero-interaction users benefit most from demographic-based approaches")
        print(f"   ⚖️ Trade-off between relevance (higher scores) and diversity (fewer categories)")
    
    def export_results(self, filename: str = None) -> None:
        """Export results to CSV for further analysis."""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zero_interaction_test_results_{timestamp}.csv"
        
        print(f"\n💾 Exporting results to {filename}...")
        
        # Flatten results for CSV
        rows = []
        for result in self.test_results:
            profile = result['profile']
            
            for algorithm, algo_result in result['algorithm_results'].items():
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'profile_name': profile['name'],
                    'profile_type': profile['profile_type'],
                    'age': profile['age'],
                    'gender': profile['gender'],
                    'income': profile['income'],
                    'profession': profile['profession'],
                    'location': profile['location'],
                    'education_level': profile['education_level'],
                    'marital_status': profile['marital_status'],
                    'interaction_count': len(profile['interaction_history']),
                    'algorithm': algorithm,
                    'success': algo_result['success'],
                    'recommendation_count': algo_result['recommendation_count'],
                    'mean_score': algo_result['mean_score'],
                    'std_score': algo_result['std_score'],
                    'min_score': algo_result['min_score'],
                    'max_score': algo_result['max_score'],
                    'category_diversity': algo_result['category_diversity'],
                    'top_category_1': list(algo_result['top_categories'].keys())[0] if algo_result['top_categories'] else '',
                    'top_category_1_count': list(algo_result['top_categories'].values())[0] if algo_result['top_categories'] else 0,
                    'elapsed_time': algo_result['elapsed_time'],
                    'error_message': algo_result['error_message'] or ''
                }
                rows.append(row)
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        print(f"✅ Results exported: {len(rows)} records in {filename}")
    
    def run_full_analysis(self, k: int = 10) -> None:
        """Run complete zero-interaction analysis pipeline."""
        
        print("🎯 ZERO-INTERACTION USER RECOMMENDATION ANALYSIS")
        print("=" * 60)
        print("Testing recommendation algorithms specifically for users with NO interaction history")
        print("This evaluates cold-start performance and demographic boosting effectiveness")
        
        # Run tests
        self.run_comprehensive_test(k)
        
        # Analyze results
        self.analyze_results()
        
        # Export results
        self.export_results()
        
        print(f"\n🎉 Zero-interaction analysis completed successfully!")
        print(f"   • Tested {len(self.test_results)} diverse demographic profiles")
        print(f"   • Compared {len(self.algorithms)} recommendation algorithms")
        print(f"   • Focus on cold-start user scenarios")
        print(f"   • Results exported for further analysis")


def main():
    """Run zero-interaction recommendation testing."""
    
    try:
        tester = ZeroInteractionTester()
        tester.run_full_analysis(k=10)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()