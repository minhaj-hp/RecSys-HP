#!/usr/bin/env python3
"""
Demographics Evaluation Script for Recommendation System

This script analyzes how well demographics are being utilized in the recommendation system,
specifically focusing on users with no interaction history (cold start problem).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from src.inference.recommendation_engine import RecommendationEngine

class DemographicsEvaluator:
    """Evaluate demographic influence in recommendation system."""
    
    def __init__(self):
        print("Loading recommendation engine...")
        self.engine = RecommendationEngine()
        self.users_df = pd.read_csv('datasets/users.csv')
        print(f"Loaded {len(self.users_df)} users from dataset")
        
        # Test user profiles with varying demographics
        self.test_profiles = self._create_test_profiles()
        
    def _create_test_profiles(self) -> List[Dict]:
        """Create diverse test user profiles for evaluation."""
        profiles = []
        
        # Profile variations
        ages = [22, 28, 35, 45, 55, 65]
        genders = ['male', 'female']
        incomes = [30000, 50000, 75000, 100000, 150000]
        professions = ['Technology', 'Healthcare', 'Education', 'Finance', 'Retail']
        locations = ['Urban', 'Suburban', 'Rural']
        educations = ['High School', "Bachelor's", "Master's", 'PhD+']
        marital_statuses = ['Single', 'Married', 'Divorced']
        
        # Create systematic combinations
        profile_id = 0
        for age in ages:
            for gender in genders:
                for income in incomes[:3]:  # Limit combinations
                    for profession in professions[:3]:
                        profile = {
                            'profile_id': profile_id,
                            'age': age,
                            'gender': gender,
                            'income': income,
                            'profession': profession,
                            'location': locations[profile_id % len(locations)],
                            'education_level': educations[profile_id % len(educations)],
                            'marital_status': marital_statuses[profile_id % len(marital_statuses)],
                            'interaction_history': []  # Zero interactions
                        }
                        profiles.append(profile)
                        profile_id += 1
        
        print(f"Created {len(profiles)} test profiles")
        return profiles
    
    def evaluate_demographic_influence(self) -> Dict:
        """Evaluate how much demographics influence recommendations."""
        print("\n=== Evaluating Demographic Influence ===")
        
        results = {
            'embedding_analysis': {},
            'recommendation_diversity': {},
            'demographic_sensitivity': {},
            'cold_start_performance': {}
        }
        
        # 1. Analyze user embedding diversity
        results['embedding_analysis'] = self._analyze_embedding_diversity()
        
        # 2. Test recommendation diversity across demographics
        results['recommendation_diversity'] = self._analyze_recommendation_diversity()
        
        # 3. Test demographic sensitivity
        results['demographic_sensitivity'] = self._test_demographic_sensitivity()
        
        # 4. Cold start performance
        results['cold_start_performance'] = self._evaluate_cold_start()
        
        return results
    
    def _analyze_embedding_diversity(self) -> Dict:
        """Analyze diversity in user embeddings for zero-interaction users."""
        print("Analyzing user embedding diversity...")
        
        embeddings = []
        profile_info = []
        
        # Get embeddings for sample profiles
        sample_profiles = self.test_profiles[:50]  # Use first 50 for speed
        
        for profile in sample_profiles:
            embedding = self.engine.get_user_embedding_enhanced(
                age=profile['age'],
                gender=profile['gender'],
                income=profile['income'],
                profession=profile['profession'],
                location=profile['location'],
                education_level=profile['education_level'],
                marital_status=profile['marital_status'],
                interaction_history=[]
            )
            embeddings.append(embedding)
            profile_info.append(profile)
        
        embeddings = np.array(embeddings)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # Analyze demographic grouping
        same_age_sims = []
        diff_age_sims = []
        same_gender_sims = []
        diff_gender_sims = []
        same_profession_sims = []
        diff_profession_sims = []
        
        for i, profile_i in enumerate(profile_info):
            for j, profile_j in enumerate(profile_info[i+1:], i+1):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                # Age similarity
                if abs(profile_i['age'] - profile_j['age']) <= 5:
                    same_age_sims.append(sim)
                else:
                    diff_age_sims.append(sim)
                
                # Gender similarity
                if profile_i['gender'] == profile_j['gender']:
                    same_gender_sims.append(sim)
                else:
                    diff_gender_sims.append(sim)
                
                # Profession similarity
                if profile_i['profession'] == profile_j['profession']:
                    same_profession_sims.append(sim)
                else:
                    diff_profession_sims.append(sim)
        
        return {
            'overall_similarity_stats': {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            },
            'demographic_clustering': {
                'same_age_similarity': {
                    'mean': np.mean(same_age_sims) if same_age_sims else 0,
                    'count': len(same_age_sims)
                },
                'diff_age_similarity': {
                    'mean': np.mean(diff_age_sims) if diff_age_sims else 0,
                    'count': len(diff_age_sims)
                },
                'same_gender_similarity': {
                    'mean': np.mean(same_gender_sims) if same_gender_sims else 0,
                    'count': len(same_gender_sims)
                },
                'diff_gender_similarity': {
                    'mean': np.mean(diff_gender_sims) if diff_gender_sims else 0,
                    'count': len(diff_gender_sims)
                },
                'same_profession_similarity': {
                    'mean': np.mean(same_profession_sims) if same_profession_sims else 0,
                    'count': len(same_profession_sims)
                },
                'diff_profession_similarity': {
                    'mean': np.mean(diff_profession_sims) if diff_profession_sims else 0,
                    'count': len(diff_profession_sims)
                }
            }
        }
    
    def _analyze_recommendation_diversity(self) -> Dict:
        """Analyze recommendation diversity across different demographic groups."""
        print("Analyzing recommendation diversity across demographics...")
        
        # Group profiles by demographics
        demographic_groups = {
            'age_groups': defaultdict(list),
            'gender_groups': defaultdict(list),
            'income_groups': defaultdict(list),
            'profession_groups': defaultdict(list)
        }
        
        # Sample profiles for analysis
        sample_profiles = self.test_profiles[:30]
        
        for profile in sample_profiles:
            # Age groups
            age_group = f"{(profile['age']//10)*10}s"
            demographic_groups['age_groups'][age_group].append(profile)
            
            # Gender groups
            demographic_groups['gender_groups'][profile['gender']].append(profile)
            
            # Income groups
            income_group = f"{profile['income']//25000*25000}-{(profile['income']//25000+1)*25000}"
            demographic_groups['income_groups'][income_group].append(profile)
            
            # Profession groups
            demographic_groups['profession_groups'][profile['profession']].append(profile)
        
        diversity_results = {}
        
        for group_type, groups in demographic_groups.items():
            diversity_results[group_type] = {}
            
            for group_name, profiles in groups.items():
                if len(profiles) < 2:
                    continue
                
                # Get recommendations for each profile in group
                group_recommendations = []
                for profile in profiles:
                    try:
                        recs = self.engine.recommend_items_collaborative(
                            age=profile['age'],
                            gender=profile['gender'],
                            income=profile['income'],
                            profession=profile['profession'],
                            location=profile['location'],
                            education_level=profile['education_level'],
                            marital_status=profile['marital_status'],
                            interaction_history=[],
                            k=10
                        )
                        rec_items = [item_id for item_id, _, _ in recs]
                        group_recommendations.append(set(rec_items))
                    except Exception as e:
                        print(f"Error getting recommendations for profile: {e}")
                        continue
                
                if len(group_recommendations) >= 2:
                    # Calculate intra-group diversity
                    overlaps = []
                    for i in range(len(group_recommendations)):
                        for j in range(i+1, len(group_recommendations)):
                            intersection = len(group_recommendations[i] & group_recommendations[j])
                            union = len(group_recommendations[i] | group_recommendations[j])
                            overlap = intersection / union if union > 0 else 0
                            overlaps.append(overlap)
                    
                    diversity_results[group_type][group_name] = {
                        'mean_overlap': np.mean(overlaps),
                        'profiles_count': len(profiles),
                        'successful_recommendations': len(group_recommendations)
                    }
        
        return diversity_results
    
    def _test_demographic_sensitivity(self) -> Dict:
        """Test how sensitive recommendations are to demographic changes."""
        print("Testing demographic sensitivity...")
        
        # Base profile
        base_profile = {
            'age': 30,
            'gender': 'male',
            'income': 75000,
            'profession': 'Technology',
            'location': 'Urban',
            'education_level': "Bachelor's",
            'marital_status': 'Single'
        }
        
        # Get base recommendations
        base_recs = self.engine.recommend_items_collaborative(**base_profile, interaction_history=[], k=10)
        base_items = set([item_id for item_id, _, _ in base_recs])
        
        sensitivity_results = {}
        
        # Test variations
        variations = {
            'age_variation': [25, 35, 45, 55],
            'income_variation': [40000, 60000, 100000, 150000],
            'profession_variation': ['Healthcare', 'Education', 'Finance', 'Retail'],
            'gender_variation': ['female'],
            'location_variation': ['Suburban', 'Rural'],
            'education_level_variation': ['High School', "Master's", 'PhD+'],
            'marital_status_variation': ['Married', 'Divorced']
        }
        
        for variation_type, values in variations.items():
            sensitivity_results[variation_type] = {}
            
            for value in values:
                # Create modified profile
                modified_profile = base_profile.copy()
                demographic_key = variation_type.replace('_variation', '')
                modified_profile[demographic_key] = value
                
                try:
                    # Get recommendations for modified profile
                    modified_recs = self.engine.recommend_items_collaborative(
                        **modified_profile, interaction_history=[], k=10
                    )
                    modified_items = set([item_id for item_id, _, _ in modified_recs])
                    
                    # Calculate similarity
                    intersection = len(base_items & modified_items)
                    union = len(base_items | modified_items)
                    jaccard_similarity = intersection / union if union > 0 else 0
                    
                    sensitivity_results[variation_type][str(value)] = {
                        'jaccard_similarity': jaccard_similarity,
                        'common_items': intersection,
                        'total_unique_items': union
                    }
                
                except Exception as e:
                    print(f"Error testing {variation_type} = {value}: {e}")
                    sensitivity_results[variation_type][str(value)] = {
                        'error': str(e)
                    }
        
        return sensitivity_results
    
    def _evaluate_cold_start(self) -> Dict:
        """Evaluate cold start recommendation performance."""
        print("Evaluating cold start performance...")
        
        # Sample real users with interactions for comparison
        users_with_history = self.users_df[self.users_df['user_id'].isin([1, 2, 3, 4, 5])]
        
        cold_start_results = {
            'zero_interaction_performance': {},
            'comparison_with_history': {}
        }
        
        # Test zero interaction users
        zero_interaction_profiles = self.test_profiles[:10]
        
        recommendation_scores = []
        recommendation_diversity = []
        
        for i, profile in enumerate(zero_interaction_profiles):
            try:
                recs = self.engine.recommend_items_collaborative(
                    age=profile['age'],
                    gender=profile['gender'],
                    income=profile['income'],
                    profession=profile['profession'],
                    location=profile['location'],
                    education_level=profile['education_level'],
                    marital_status=profile['marital_status'],
                    interaction_history=[],
                    k=10
                )
                
                if recs:
                    scores = [score for _, score, _ in recs]
                    recommendation_scores.extend(scores)
                    
                    # Check category diversity
                    categories = [info.get('category_code', '').split('.')[0] 
                                for _, _, info in recs]
                    unique_categories = len(set(categories))
                    recommendation_diversity.append(unique_categories)
                
            except Exception as e:
                print(f"Error in cold start evaluation for profile {i}: {e}")
        
        cold_start_results['zero_interaction_performance'] = {
            'mean_recommendation_score': np.mean(recommendation_scores) if recommendation_scores else 0,
            'std_recommendation_score': np.std(recommendation_scores) if recommendation_scores else 0,
            'mean_category_diversity': np.mean(recommendation_diversity) if recommendation_diversity else 0,
            'profiles_tested': len(zero_interaction_profiles)
        }
        
        return cold_start_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = """
# Demographics Evaluation Report

## Executive Summary
This report analyzes the role of demographics in the recommendation system,
particularly for users with no interaction history (cold start scenario).

## Key Findings

### 1. User Embedding Diversity Analysis
"""
        
        embedding_stats = results['embedding_analysis']['overall_similarity_stats']
        report += f"""
- **Overall Embedding Similarity**: Mean = {embedding_stats['mean']:.4f}, Std = {embedding_stats['std']:.4f}
- **Similarity Range**: [{embedding_stats['min']:.4f}, {embedding_stats['max']:.4f}]
"""
        
        demographic_clustering = results['embedding_analysis']['demographic_clustering']
        report += f"""
- **Same Age vs Different Age**: {demographic_clustering['same_age_similarity']['mean']:.4f} vs {demographic_clustering['diff_age_similarity']['mean']:.4f}
- **Same Gender vs Different Gender**: {demographic_clustering['same_gender_similarity']['mean']:.4f} vs {demographic_clustering['diff_gender_similarity']['mean']:.4f}
- **Same Profession vs Different Profession**: {demographic_clustering['same_profession_similarity']['mean']:.4f} vs {demographic_clustering['diff_profession_similarity']['mean']:.4f}
"""
        
        report += "\n### 2. Recommendation Diversity Analysis\n"
        
        diversity = results['recommendation_diversity']
        for group_type, groups in diversity.items():
            report += f"\n**{group_type.replace('_', ' ').title()}:**\n"
            for group_name, stats in groups.items():
                report += f"- {group_name}: {stats['mean_overlap']:.4f} overlap (n={stats['profiles_count']})\n"
        
        report += "\n### 3. Demographic Sensitivity Analysis\n"
        
        sensitivity = results['demographic_sensitivity']
        for variation_type, variations in sensitivity.items():
            report += f"\n**{variation_type.replace('_', ' ').title()}:**\n"
            for value, stats in variations.items():
                if 'error' not in stats:
                    report += f"- {value}: {stats['jaccard_similarity']:.4f} similarity, {stats['common_items']} common items\n"
        
        report += "\n### 4. Cold Start Performance\n"
        
        cold_start = results['cold_start_performance']['zero_interaction_performance']
        report += f"""
- **Mean Recommendation Score**: {cold_start['mean_recommendation_score']:.4f} ± {cold_start['std_recommendation_score']:.4f}
- **Mean Category Diversity**: {cold_start['mean_category_diversity']:.2f} categories per recommendation set
- **Profiles Successfully Tested**: {cold_start['profiles_tested']}
"""
        
        report += """

## Conclusions

### Demographics Impact Assessment:
1. **Low Impact**: If embeddings show high similarity across different demographics
2. **Medium Impact**: If some demographic variations show meaningful differences  
3. **High Impact**: If demographics create clear clustering in embedding space

### Recommendations:
- If similarity is > 0.9 across demographics: **Critical - Demographics have minimal impact**
- If overlap is > 0.8 within demographic groups: **Major issue - Insufficient personalization**
- If sensitivity scores are > 0.8: **Problem - Recommendations not responsive to demographic changes**

## Next Steps:
1. Implement stronger demographic feature engineering
2. Consider model architecture changes for better demographic utilization
3. Add demographic-specific training objectives
4. Implement demographic clustering for cold-start scenarios
"""
        
        return report
    
    def save_results(self, results: Dict, report: str):
        """Save evaluation results and report."""
        
        # Save detailed results as JSON
        with open('demographics_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        with open('demographics_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print("\nResults saved:")
        print("- demographics_evaluation_results.json (detailed data)")
        print("- demographics_evaluation_report.txt (human-readable report)")

def main():
    """Run demographics evaluation."""
    print("=== Demographics Evaluation for Recommendation System ===")
    
    try:
        # Initialize evaluator
        evaluator = DemographicsEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_demographic_influence()
        
        # Generate report
        report = evaluator.generate_report(results)
        
        # Print key findings
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # Save results
        evaluator.save_results(results, report)
        
        print("\n✅ Demographics evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()