#!/usr/bin/env python3
"""
Test 50% Category Alignment for Demographic Recommendations

This script specifically tests whether our demographic clustering system
is providing approximately 50% category alignment as expected for
users with different demographic profiles.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# Add src to path
sys.path.append('src')
from src.inference.recommendation_engine import RecommendationEngine
from src.inference.demographic_clustering import DemographicClusterer

class CategoryAlignmentTester:
    """Test category alignment in demographic recommendations."""
    
    def __init__(self):
        print("🔧 Initializing Category Alignment Tester...")
        self.engine = RecommendationEngine()
        self.clusterer = DemographicClusterer()
        
        # Test profiles for category alignment testing
        self.test_profiles = self._create_test_profiles()
        
        print(f"✅ Loaded {len(self.test_profiles)} test profiles for category alignment testing")
    
    def _create_test_profiles(self) -> List[Dict]:
        """Create test profiles with known demographic characteristics."""
        profiles = [
            # Young Tech Workers - Should prefer electronics, computers
            {"name": "YoungTech1", "age": 25, "gender": "male", "income": 85000, 
             "profession": "Technology", "location": "Urban", "education_level": "Bachelor's", "marital_status": "Single"},
            {"name": "YoungTech2", "age": 27, "gender": "female", "income": 78000,
             "profession": "Technology", "location": "Urban", "education_level": "Master's", "marital_status": "Single"},
            
            # Healthcare Workers - Should prefer appliances, smartphones  
            {"name": "Healthcare1", "age": 38, "gender": "female", "income": 65000,
             "profession": "Healthcare", "location": "Suburban", "education_level": "Bachelor's", "marital_status": "Married"},
            {"name": "Healthcare2", "age": 42, "gender": "male", "income": 72000,
             "profession": "Healthcare", "location": "Urban", "education_level": "Master's", "marital_status": "Married"},
            
            # Finance Professionals - Should prefer high-end electronics
            {"name": "Finance1", "age": 45, "gender": "female", "income": 95000,
             "profession": "Finance", "location": "Urban", "education_level": "Bachelor's", "marital_status": "Divorced"},
            {"name": "Finance2", "age": 55, "gender": "male", "income": 135000,
             "profession": "Finance", "location": "Urban", "education_level": "Master's", "marital_status": "Married"},
            
            # Education Workers - Should prefer electronics, computers
            {"name": "Education1", "age": 28, "gender": "female", "income": 45000,
             "profession": "Education", "location": "Suburban", "education_level": "Master's", "marital_status": "Single"},
            {"name": "Education2", "age": 40, "gender": "male", "income": 52000,
             "profession": "Education", "location": "Rural", "education_level": "PhD+", "marital_status": "Married"},
            
            # Retail Workers - Should prefer affordable items, apparel
            {"name": "Retail1", "age": 24, "gender": "female", "income": 32000,
             "profession": "Retail", "location": "Urban", "education_level": "High School", "marital_status": "Single"},
            {"name": "Retail2", "age": 35, "gender": "male", "income": 38000,
             "profession": "Retail", "location": "Suburban", "education_level": "Some College", "marital_status": "Married"},
        ]
        
        # Add zero interaction history to all profiles
        for profile in profiles:
            profile['interaction_history'] = []
            
        return profiles
    
    def get_cluster_preferences(self, profile: Dict) -> List[Tuple[str, float]]:
        """Get demographic cluster preferences for a profile."""
        try:
            preferences = self.clusterer.get_category_recommendations_for_user(
                age=profile['age'],
                gender=profile['gender'], 
                income=profile['income'],
                profession=profile['profession'],
                location=profile['location'],
                education_level=profile['education_level'],
                marital_status=profile['marital_status'],
                top_k=10
            )
            return preferences
        except Exception as e:
            print(f"Error getting cluster preferences for {profile['name']}: {e}")
            return []
    
    def test_category_alignment_for_profile(self, profile: Dict, k: int = 20) -> Dict:
        """Test category alignment for a single profile."""
        
        print(f"\n🧪 Testing Category Alignment: {profile['name']}")
        print(f"   Demographics: {profile['age']}y {profile['gender']} ${profile['income']} {profile['profession']}")
        
        # Get cluster preferences
        cluster_preferences = self.get_cluster_preferences(profile)
        if not cluster_preferences:
            return {"success": False, "error": "No cluster preferences found"}
        
        preferred_categories = set([cat for cat, _ in cluster_preferences[:5]])
        print(f"   Preferred Categories: {list(preferred_categories)}")
        
        # Get demographic-boosted recommendations
        try:
            recommendations = self.engine.recommend_items_demographic_boosted(
                age=profile['age'],
                gender=profile['gender'],
                income=profile['income'],
                profession=profile['profession'],
                location=profile['location'],
                education_level=profile['education_level'],
                marital_status=profile['marital_status'],
                interaction_history=[],
                k=k
            )
            
            if not recommendations:
                return {"success": False, "error": "No recommendations returned"}
            
            # Analyze category alignment
            rec_categories = []
            aligned_count = 0
            
            for item_id, score, item_info in recommendations:
                item_category = item_info.get('category_code', '')
                rec_categories.append(item_category)
                
                # Check if item category aligns with preferred categories
                is_aligned = False
                for pref_cat in preferred_categories:
                    if item_category.startswith(pref_cat):
                        is_aligned = True
                        break
                
                if is_aligned:
                    aligned_count += 1
            
            # Calculate alignment percentage
            alignment_percentage = (aligned_count / len(recommendations)) * 100
            
            # Category distribution analysis
            category_dist = Counter()
            for cat in rec_categories:
                if '.' in cat:
                    top_level = cat.split('.')[0]
                else:
                    top_level = cat
                category_dist[top_level] += 1
            
            result = {
                "success": True,
                "profile_name": profile['name'],
                "total_recommendations": len(recommendations),
                "aligned_recommendations": aligned_count,
                "alignment_percentage": alignment_percentage,
                "preferred_categories": list(preferred_categories),
                "recommendation_categories": rec_categories,
                "category_distribution": dict(category_dist),
                "cluster_preferences": cluster_preferences[:5]
            }
            
            print(f"   📊 Alignment: {aligned_count}/{len(recommendations)} ({alignment_percentage:.1f}%)")
            print(f"   📂 Category Distribution: {dict(category_dist)}")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_all_profiles(self, k: int = 20) -> List[Dict]:
        """Test category alignment for all profiles."""
        
        print("="*80)
        print("🎯 TESTING 50% CATEGORY ALIGNMENT FOR DEMOGRAPHIC RECOMMENDATIONS")
        print("="*80)
        
        results = []
        
        for profile in self.test_profiles:
            result = self.test_category_alignment_for_profile(profile, k=k)
            if result["success"]:
                results.append(result)
        
        return results
    
    def analyze_alignment_results(self, results: List[Dict]):
        """Analyze category alignment results across all profiles."""
        
        print("\n" + "="*80)
        print("📊 CATEGORY ALIGNMENT ANALYSIS")
        print("="*80)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Overall statistics
        alignment_percentages = [r["alignment_percentage"] for r in results]
        
        print(f"\n🎯 OVERALL ALIGNMENT STATISTICS:")
        print(f"   Profiles Tested: {len(results)}")
        print(f"   Average Alignment: {np.mean(alignment_percentages):.1f}%")
        print(f"   Median Alignment: {np.median(alignment_percentages):.1f}%")
        print(f"   Std Deviation: {np.std(alignment_percentages):.1f}%")
        print(f"   Min Alignment: {np.min(alignment_percentages):.1f}%")
        print(f"   Max Alignment: {np.max(alignment_percentages):.1f}%")
        
        # Target analysis (50% alignment)
        target_alignment = 50.0
        profiles_meeting_target = sum(1 for p in alignment_percentages if p >= target_alignment)
        
        print(f"\n🎯 TARGET ALIGNMENT (50%) ANALYSIS:")
        print(f"   Profiles Meeting Target: {profiles_meeting_target}/{len(results)} ({profiles_meeting_target/len(results)*100:.1f}%)")
        
        # Detailed breakdown by profession
        profession_results = defaultdict(list)
        for result in results:
            # Extract profession from profile name
            profile_name = result["profile_name"]
            if "Tech" in profile_name:
                profession = "Technology"
            elif "Healthcare" in profile_name:
                profession = "Healthcare"
            elif "Finance" in profile_name:
                profession = "Finance"
            elif "Education" in profile_name:
                profession = "Education"
            elif "Retail" in profile_name:
                profession = "Retail"
            else:
                profession = "Other"
            
            profession_results[profession].append(result["alignment_percentage"])
        
        print(f"\n📈 ALIGNMENT BY PROFESSION:")
        for profession, alignments in profession_results.items():
            avg_alignment = np.mean(alignments)
            print(f"   {profession}: {avg_alignment:.1f}% average ({len(alignments)} profiles)")
        
        # Detailed profile results
        print(f"\n📋 DETAILED PROFILE RESULTS:")
        for result in results:
            status = "✅ PASS" if result["alignment_percentage"] >= target_alignment else "⚠️  BELOW TARGET"
            print(f"   {result['profile_name']}: {result['alignment_percentage']:.1f}% alignment {status}")
            
            # Show top preferred vs actual categories
            pref_cats = [cat.split('.')[0] for cat in result['preferred_categories']]
            actual_dist = result['category_distribution']
            
            print(f"      Preferred: {pref_cats}")
            print(f"      Actual: {actual_dist}")
        
        # Recommendations
        print(f"\n🔧 RECOMMENDATIONS:")
        avg_alignment = np.mean(alignment_percentages)
        
        if avg_alignment >= 45.0:
            print(f"   ✅ EXCELLENT: Average {avg_alignment:.1f}% alignment is close to 50% target")
        elif avg_alignment >= 35.0:
            print(f"   ⚠️  GOOD: Average {avg_alignment:.1f}% alignment is reasonable but could be improved")
            print(f"      Consider: Stronger category boosting multipliers")
        elif avg_alignment >= 25.0:
            print(f"   ⚠️  FAIR: Average {avg_alignment:.1f}% alignment shows some demographic influence")
            print(f"      Consider: Reviewing cluster preferences and boosting logic")
        else:
            print(f"   ❌ POOR: Average {avg_alignment:.1f}% alignment suggests demographic clustering issues")
            print(f"      Action: Debug cluster formation and preference learning")
    
    def run_comprehensive_test(self, k: int = 20):
        """Run comprehensive category alignment testing."""
        
        print("🚀 Starting Category Alignment Testing")
        print(f"   Testing {k} recommendations per profile")
        print(f"   Target: 50% category alignment with demographic preferences")
        
        # Test all profiles
        results = self.test_all_profiles(k=k)
        
        # Analyze results
        self.analyze_alignment_results(results)
        
        print("\n🎉 Category Alignment Testing Completed!")


def main():
    """Run category alignment testing."""
    
    try:
        tester = CategoryAlignmentTester()
        tester.run_comprehensive_test(k=20)  # Test with 20 recommendations for better statistics
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()