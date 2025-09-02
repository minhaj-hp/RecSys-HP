#!/usr/bin/env python3
"""
Comprehensive Demographics Recommendations Testing Script

This script tests demographic-based recommendations for users with and without
interaction history, validating the effectiveness of our demographic enhancements.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import csv
from collections import Counter, defaultdict
import time
from datetime import datetime

# Add src to path
sys.path.append('src')
from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector

class DemographicsRecommendationTester:
    """Test demographic recommendations comprehensively."""
    
    def __init__(self):
        print("🔧 Initializing Demographics Recommendation Tester...")
        self.engine = RecommendationEngine()
        self.real_user_selector = RealUserSelector()
        self.test_results = []
        
        # Test user profiles for zero-interaction testing
        self.zero_interaction_profiles = self._create_zero_interaction_profiles()
        
        print(f"✅ Loaded {len(self.zero_interaction_profiles)} zero-interaction test profiles")
    
    def _create_zero_interaction_profiles(self) -> List[Dict]:
        """Create diverse zero-interaction test profiles."""
        profiles = []
        
        # Define test segments
        segments = [
            # Young Tech Workers
            {"name": "Young_Tech_Male_Urban_High", "age": 25, "gender": "male", "income": 85000, 
             "profession": "Technology", "location": "Urban", "education_level": "Bachelor's", "marital_status": "Single"},
            {"name": "Young_Tech_Female_Urban_High", "age": 27, "gender": "female", "income": 78000,
             "profession": "Technology", "location": "Urban", "education_level": "Master's", "marital_status": "Single"},
            
            # Healthcare Workers
            {"name": "Mid_Healthcare_Female_Sub_Med", "age": 38, "gender": "female", "income": 65000,
             "profession": "Healthcare", "location": "Suburban", "education_level": "Bachelor's", "marital_status": "Married"},
            {"name": "Mid_Healthcare_Male_Urban_Med", "age": 42, "gender": "male", "income": 72000,
             "profession": "Healthcare", "location": "Urban", "education_level": "Master's", "marital_status": "Married"},
            
            # Finance Professionals  
            {"name": "Senior_Finance_Male_Urban_VHigh", "age": 55, "gender": "male", "income": 135000,
             "profession": "Finance", "location": "Urban", "education_level": "Master's", "marital_status": "Married"},
            {"name": "Mid_Finance_Female_Urban_High", "age": 45, "gender": "female", "income": 95000,
             "profession": "Finance", "location": "Urban", "education_level": "Bachelor's", "marital_status": "Divorced"},
            
            # Education Sector
            {"name": "Young_Education_Female_Sub_Low", "age": 28, "gender": "female", "income": 45000,
             "profession": "Education", "location": "Suburban", "education_level": "Master's", "marital_status": "Single"},
            {"name": "Mid_Education_Male_Rural_Med", "age": 40, "gender": "male", "income": 52000,
             "profession": "Education", "location": "Rural", "education_level": "PhD+", "marital_status": "Married"},
            
            # Retail Workers
            {"name": "Young_Retail_Female_Urban_Low", "age": 24, "gender": "female", "income": 32000,
             "profession": "Retail", "location": "Urban", "education_level": "High School", "marital_status": "Single"},
            {"name": "Mid_Retail_Male_Sub_Low", "age": 35, "gender": "male", "income": 38000,
             "profession": "Retail", "location": "Suburban", "education_level": "Some College", "marital_status": "Married"},
            
            # Diverse Demographics for Edge Cases
            {"name": "Senior_Female_Rural_Low", "age": 62, "gender": "female", "income": 28000,
             "profession": "Other", "location": "Rural", "education_level": "High School", "marital_status": "Widowed"},
            {"name": "Young_Male_Urban_Student", "age": 20, "gender": "male", "income": 15000,
             "profession": "Other", "location": "Urban", "education_level": "Some College", "marital_status": "Single"}
        ]
        
        for segment in segments:
            profile = segment.copy()
            profile['interaction_history'] = []  # Zero interactions
            profile['profile_type'] = 'zero_interaction'
            profiles.append(profile)
        
        return profiles
    
    def get_real_users_with_history(self, count: int = 6) -> List[Dict]:
        """Get real users with interaction history for testing."""
        print(f"📊 Getting {count} real users with interaction history...")
        
        real_users = self.real_user_selector.get_real_users(n=count, min_interactions=10)
        
        formatted_users = []
        for user in real_users:
            formatted_user = {
                'name': f"Real_User_{user['user_id']}",
                'user_id': user['user_id'],
                'age': user['age'],
                'gender': user['gender'],
                'income': user['income'],
                'profession': user.get('profession', 'Other'),
                'location': user.get('location', 'Urban'),
                'education_level': user.get('education_level', 'Bachelor\'s'),
                'marital_status': user.get('marital_status', 'Single'),
                'interaction_history': user['interaction_history'][:20],  # Limit to 20 most recent
                'profile_type': 'with_history',
                'interaction_count': len(user['interaction_history'])
            }
            formatted_users.append(formatted_user)
        
        print(f"✅ Retrieved {len(formatted_users)} real users")
        return formatted_users
    
    def test_recommendation_method(self, profile: Dict, method: str, k: int = 10) -> Dict:
        """Test a specific recommendation method for a profile."""
        
        try:
            start_time = time.time()
            
            if method == "collaborative":
                recommendations = self.engine.recommend_items_collaborative(
                    age=profile['age'],
                    gender=profile['gender'], 
                    income=profile['income'],
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],
                    k=k
                )
            
            elif method == "demographic_boosted":
                recommendations = self.engine.recommend_items_demographic_boosted(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'], 
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],
                    k=k
                )
            
            elif method == "hybrid":
                recommendations = self.engine.recommend_items_hybrid(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'],
                    profession=profile['profession'], 
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=profile['interaction_history'],
                    k=k
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed_time = time.time() - start_time
            
            # Analyze recommendations
            if recommendations:
                scores = [score for _, score, _ in recommendations]
                categories = [info.get('category_code', '').split('.')[0] for _, _, info in recommendations]
                category_diversity = len(set(categories))
                
                # Get top-level categories
                top_categories = Counter(categories).most_common(3)
                
                return {
                    'success': True,
                    'method': method,
                    'count': len(recommendations),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'category_diversity': category_diversity,
                    'top_categories': top_categories,
                    'elapsed_time': elapsed_time,
                    'recommendations': recommendations[:5]  # Store first 5 for analysis
                }
            else:
                return {
                    'success': False,
                    'method': method,
                    'error': 'No recommendations returned',
                    'elapsed_time': elapsed_time
                }
        
        except Exception as e:
            return {
                'success': False,
                'method': method, 
                'error': str(e),
                'elapsed_time': 0
            }
    
    def test_profile_comprehensive(self, profile: Dict) -> Dict:
        """Test all recommendation methods for a single profile."""
        
        profile_name = profile['name']
        has_history = len(profile.get('interaction_history', [])) > 0
        
        print(f"\n🧪 Testing Profile: {profile_name}")
        print(f"   Demographics: {profile['age']}y {profile['gender']} ${profile['income']} {profile['profession']}")
        print(f"   Location: {profile['location']}, Education: {profile['education_level']}, Marital: {profile['marital_status']}")
        print(f"   Interaction History: {len(profile.get('interaction_history', []))} items")
        
        # Test different methods
        methods = ["collaborative", "demographic_boosted", "hybrid"]
        if not has_history:
            print("   🔥 ZERO-INTERACTION USER - Testing demographic personalization")
        
        results = {}
        for method in methods:
            print(f"      → Testing {method}...")
            result = self.test_recommendation_method(profile, method)
            results[method] = result
            
            if result['success']:
                print(f"         ✅ {result['count']} recs, diversity: {result['category_diversity']}, score: {result['mean_score']:.4f}")
                print(f"         📂 Top categories: {[f'{cat}({count})' for cat, count in result['top_categories']]}")
            else:
                print(f"         ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Create comprehensive result
        test_result = {
            'timestamp': datetime.now().isoformat(),
            'profile_name': profile_name,
            'profile_type': profile.get('profile_type', 'unknown'),
            'demographics': {
                'age': profile['age'],
                'gender': profile['gender'],
                'income': profile['income'],
                'profession': profile['profession'],
                'location': profile['location'],
                'education_level': profile['education_level'],
                'marital_status': profile['marital_status']
            },
            'interaction_count': len(profile.get('interaction_history', [])),
            'has_history': has_history,
            'method_results': results
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def test_zero_interaction_users(self):
        """Test all zero-interaction user profiles."""
        print("\n" + "="*80)
        print("🚀 TESTING ZERO-INTERACTION USERS (Cold Start)")
        print("="*80)
        
        zero_interaction_results = []
        
        for profile in self.zero_interaction_profiles:
            result = self.test_profile_comprehensive(profile)
            zero_interaction_results.append(result)
        
        # Analyze zero-interaction results
        print(f"\n📈 ZERO-INTERACTION ANALYSIS:")
        print(f"   Profiles Tested: {len(zero_interaction_results)}")
        
        # Success rates by method
        methods = ["collaborative", "demographic_boosted", "hybrid"]
        for method in methods:
            success_count = sum(1 for r in zero_interaction_results 
                              if r['method_results'][method]['success'])
            print(f"   {method} Success Rate: {success_count}/{len(zero_interaction_results)} ({success_count/len(zero_interaction_results)*100:.1f}%)")
            
            if success_count > 0:
                diversities = [r['method_results'][method]['category_diversity'] 
                             for r in zero_interaction_results 
                             if r['method_results'][method]['success']]
                scores = [r['method_results'][method]['mean_score']
                         for r in zero_interaction_results
                         if r['method_results'][method]['success']]
                
                print(f"      Average Category Diversity: {np.mean(diversities):.2f}")
                print(f"      Average Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return zero_interaction_results
    
    def test_users_with_history(self):
        """Test users with interaction history."""
        print("\n" + "="*80)
        print("📚 TESTING USERS WITH INTERACTION HISTORY")
        print("="*80)
        
        users_with_history = self.get_real_users_with_history(count=6)
        history_results = []
        
        for profile in users_with_history:
            result = self.test_profile_comprehensive(profile)
            history_results.append(result)
        
        # Analyze results with history
        print(f"\n📊 USERS-WITH-HISTORY ANALYSIS:")
        print(f"   Profiles Tested: {len(history_results)}")
        
        # Success rates and performance by method
        methods = ["collaborative", "demographic_boosted", "hybrid"]
        for method in methods:
            success_count = sum(1 for r in history_results 
                              if r['method_results'][method]['success'])
            print(f"   {method} Success Rate: {success_count}/{len(history_results)} ({success_count/len(history_results)*100:.1f}%)")
            
            if success_count > 0:
                diversities = [r['method_results'][method]['category_diversity']
                             for r in history_results
                             if r['method_results'][method]['success']]
                scores = [r['method_results'][method]['mean_score']
                         for r in history_results
                         if r['method_results'][method]['success']]
                
                print(f"      Average Category Diversity: {np.mean(diversities):.2f}")
                print(f"      Average Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return history_results
    
    def compare_demographic_sensitivity(self):
        """Test demographic sensitivity with variations."""
        print("\n" + "="*80)
        print("🔬 DEMOGRAPHIC SENSITIVITY TESTING")
        print("="*80)
        
        # Base profile for sensitivity testing
        base_profile = {
            'name': 'Sensitivity_Base',
            'age': 35,
            'gender': 'male',
            'income': 75000,
            'profession': 'Technology',
            'location': 'Urban',
            'education_level': 'Bachelor\'s',
            'marital_status': 'Single',
            'interaction_history': [],
            'profile_type': 'sensitivity_test'
        }
        
        print("🎯 Base Profile:")
        print(f"   {base_profile['age']}y {base_profile['gender']} ${base_profile['income']} {base_profile['profession']}")
        
        # Get base recommendations
        base_result = self.test_recommendation_method(base_profile, "demographic_boosted")
        if not base_result['success']:
            print("❌ Failed to get base recommendations")
            return
        
        base_items = set([item_id for item_id, _, _ in base_result['recommendations']])
        
        # Test variations
        variations = [
            # Age variations
            {'age': 25, 'variation_type': 'age', 'variation_value': '25'},
            {'age': 45, 'variation_type': 'age', 'variation_value': '45'},
            {'age': 55, 'variation_type': 'age', 'variation_value': '55'},
            
            # Income variations  
            {'income': 45000, 'variation_type': 'income', 'variation_value': '45000'},
            {'income': 100000, 'variation_type': 'income', 'variation_value': '100000'},
            {'income': 150000, 'variation_type': 'income', 'variation_value': '150000'},
            
            # Gender variation
            {'gender': 'female', 'variation_type': 'gender', 'variation_value': 'female'},
            
            # Profession variations
            {'profession': 'Healthcare', 'variation_type': 'profession', 'variation_value': 'Healthcare'},
            {'profession': 'Finance', 'variation_type': 'profession', 'variation_value': 'Finance'},
            {'profession': 'Education', 'variation_type': 'profession', 'variation_value': 'Education'},
            
            # Location variations
            {'location': 'Suburban', 'variation_type': 'location', 'variation_value': 'Suburban'},
            {'location': 'Rural', 'variation_type': 'location', 'variation_value': 'Rural'},
        ]
        
        sensitivity_results = []
        
        for variation in variations:
            # Create modified profile
            modified_profile = base_profile.copy()
            modified_profile.update(variation)
            modified_profile['name'] = f"Sensitivity_{variation['variation_type']}_{variation['variation_value']}"
            
            # Test modified profile
            modified_result = self.test_recommendation_method(modified_profile, "demographic_boosted")
            
            if modified_result['success']:
                modified_items = set([item_id for item_id, _, _ in modified_result['recommendations']])
                
                # Calculate similarity
                intersection = len(base_items & modified_items)
                union = len(base_items | modified_items)
                jaccard_similarity = intersection / union if union > 0 else 0
                
                sensitivity_result = {
                    'variation_type': variation['variation_type'],
                    'variation_value': variation['variation_value'],
                    'jaccard_similarity': jaccard_similarity,
                    'common_items': intersection,
                    'different_items': len(base_items ^ modified_items),
                    'category_diversity': modified_result['category_diversity'],
                    'mean_score': modified_result['mean_score']
                }
                
                sensitivity_results.append(sensitivity_result)
                
                print(f"   {variation['variation_type']} = {variation['variation_value']}: "
                      f"{jaccard_similarity:.3f} similarity, {intersection}/10 common items, "
                      f"{modified_result['category_diversity']} categories")
            else:
                print(f"   {variation['variation_type']} = {variation['variation_value']}: FAILED")
        
        # Summary of sensitivity
        print(f"\n📈 DEMOGRAPHIC SENSITIVITY SUMMARY:")
        for var_type in ['age', 'income', 'gender', 'profession', 'location']:
            type_results = [r for r in sensitivity_results if r['variation_type'] == var_type]
            if type_results:
                similarities = [r['jaccard_similarity'] for r in type_results]
                print(f"   {var_type}: {np.mean(similarities):.3f} ± {np.std(similarities):.3f} average similarity")
        
        return sensitivity_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE DEMOGRAPHICS TESTING REPORT")
        print("="*80)
        
        if not self.test_results:
            print("❌ No test results available")
            return
        
        # Separate results by type
        zero_interaction_results = [r for r in self.test_results if r['profile_type'] == 'zero_interaction']
        with_history_results = [r for r in self.test_results if r['profile_type'] == 'with_history']
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Profiles Tested: {len(self.test_results)}")
        print(f"   Zero-Interaction Profiles: {len(zero_interaction_results)}")
        print(f"   With-History Profiles: {len(with_history_results)}")
        
        # Method comparison
        methods = ["collaborative", "demographic_boosted", "hybrid"]
        print(f"\n🎯 METHOD COMPARISON:")
        
        for method in methods:
            print(f"\n   === {method.upper()} ===")
            
            # Zero-interaction performance
            zero_success = sum(1 for r in zero_interaction_results 
                              if r['method_results'][method]['success'])
            zero_diversities = [r['method_results'][method]['category_diversity']
                               for r in zero_interaction_results
                               if r['method_results'][method]['success']]
            zero_scores = [r['method_results'][method]['mean_score']
                          for r in zero_interaction_results
                          if r['method_results'][method]['success']]
            
            print(f"      Zero-Interaction Users:")
            print(f"        Success Rate: {zero_success}/{len(zero_interaction_results)} ({zero_success/len(zero_interaction_results)*100:.1f}%)")
            if zero_diversities:
                print(f"        Category Diversity: {np.mean(zero_diversities):.2f} ± {np.std(zero_diversities):.2f}")
                print(f"        Recommendation Score: {np.mean(zero_scores):.4f} ± {np.std(zero_scores):.4f}")
            
            # With-history performance
            if with_history_results:
                hist_success = sum(1 for r in with_history_results
                                  if r['method_results'][method]['success'])
                hist_diversities = [r['method_results'][method]['category_diversity']
                                   for r in with_history_results
                                   if r['method_results'][method]['success']]
                hist_scores = [r['method_results'][method]['mean_score']
                              for r in with_history_results
                              if r['method_results'][method]['success']]
                
                print(f"      With-History Users:")
                print(f"        Success Rate: {hist_success}/{len(with_history_results)} ({hist_success/len(with_history_results)*100:.1f}%)")
                if hist_diversities:
                    print(f"        Category Diversity: {np.mean(hist_diversities):.2f} ± {np.std(hist_diversities):.2f}")
                    print(f"        Recommendation Score: {np.mean(hist_scores):.4f} ± {np.std(hist_scores):.4f}")
        
        print(f"\n✅ DEMOGRAPHICS TESTING COMPLETED")
        print(f"   Test Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def save_results_to_csv(self, filename: str = "demographics_test_results.csv"):
        """Save all test results to CSV for further analysis."""
        
        if not self.test_results:
            print("❌ No results to save")
            return
        
        csv_rows = []
        
        for result in self.test_results:
            base_row = {
                'timestamp': result['timestamp'],
                'profile_name': result['profile_name'],
                'profile_type': result['profile_type'],
                'age': result['demographics']['age'],
                'gender': result['demographics']['gender'],
                'income': result['demographics']['income'],
                'profession': result['demographics']['profession'],
                'location': result['demographics']['location'],
                'education_level': result['demographics']['education_level'],
                'marital_status': result['demographics']['marital_status'],
                'interaction_count': result['interaction_count'],
                'has_history': result['has_history']
            }
            
            # Add method results
            for method, method_result in result['method_results'].items():
                row = base_row.copy()
                row['method'] = method
                row['success'] = method_result['success']
                
                if method_result['success']:
                    row.update({
                        'recommendation_count': method_result['count'],
                        'mean_score': method_result['mean_score'],
                        'std_score': method_result['std_score'],
                        'category_diversity': method_result['category_diversity'],
                        'top_category_1': method_result['top_categories'][0][0] if method_result['top_categories'] else '',
                        'top_category_1_count': method_result['top_categories'][0][1] if method_result['top_categories'] else 0,
                        'elapsed_time': method_result['elapsed_time']
                    })
                else:
                    row.update({
                        'error': method_result.get('error', ''),
                        'elapsed_time': method_result['elapsed_time']
                    })
                
                csv_rows.append(row)
        
        # Save to CSV
        with open(filename, 'w', newline='') as csvfile:
            if csv_rows:
                fieldnames = csv_rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
        
        print(f"💾 Results saved to {filename}")
        print(f"   {len(csv_rows)} rows written")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive Demographics Recommendation Testing")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test zero-interaction users
        zero_results = self.test_zero_interaction_users()
        
        # Test users with history
        history_results = self.test_users_with_history()
        
        # Test demographic sensitivity
        sensitivity_results = self.compare_demographic_sensitivity()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Save results
        self.save_results_to_csv()
        
        print("\n🎉 ALL TESTING COMPLETED SUCCESSFULLY!")


def main():
    """Run comprehensive demographics recommendation testing."""
    
    try:
        tester = DemographicsRecommendationTester()
        tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()