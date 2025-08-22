#!/usr/bin/env python3
"""
Enhanced recommendation engine with category-aware filtering and improved user alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector


class EnhancedRecommendationEngine(RecommendationEngine):
    """Enhanced recommendation engine with category-aware improvements."""
    
    def __init__(self, artifacts_path: str = "src/artifacts/"):
        super().__init__(artifacts_path)
        self.real_user_selector = RealUserSelector()
    
    def _analyze_user_category_preferences(self, interaction_history: List[int]) -> Dict[str, float]:
        """Analyze user's category preferences from interaction history."""
        
        if not interaction_history:
            return {}
        
        category_counts = Counter()
        
        for item_id in interaction_history:
            # Get item category from items dataframe
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if not item_row.empty:
                category = item_row.iloc[0].get('category_code', 'Unknown')
                category_counts[category] += 1
        
        # Convert to percentages
        total_interactions = sum(category_counts.values())
        if total_interactions == 0:
            return {}
        
        category_preferences = {}
        for category, count in category_counts.items():
            category_preferences[category] = count / total_interactions
        
        return category_preferences
    
    def _boost_category_aligned_recommendations(self, 
                                              recommendations: List[Tuple[int, float, Dict]],
                                              user_category_preferences: Dict[str, float],
                                              boost_factor: float = 1.5) -> List[Tuple[int, float, Dict]]:
        """Boost recommendations that align with user's category preferences."""
        
        if not user_category_preferences:
            return recommendations
        
        boosted_recs = []
        
        for item_id, score, item_info in recommendations:
            item_category = item_info.get('category_code', 'Unknown')
            
            # Apply category boost if user has preference for this category
            category_preference = user_category_preferences.get(item_category, 0)
            
            if category_preference > 0:
                # Boost score based on user's preference strength
                boosted_score = score * (1 + boost_factor * category_preference)
                boosted_recs.append((item_id, boosted_score, item_info))
            else:
                boosted_recs.append((item_id, score, item_info))
        
        # Re-sort by boosted scores
        boosted_recs.sort(key=lambda x: x[1], reverse=True)
        return boosted_recs
    
    def _diversify_recommendations(self,
                                  recommendations: List[Tuple[int, float, Dict]],
                                  max_per_category: int = 3) -> List[Tuple[int, float, Dict]]:
        """Ensure category diversity in recommendations."""
        
        category_counts = Counter()
        diversified_recs = []
        
        for item_id, score, item_info in recommendations:
            item_category = item_info.get('category_code', 'Unknown')
            
            if category_counts[item_category] < max_per_category:
                diversified_recs.append((item_id, score, item_info))
                category_counts[item_category] += 1
        
        return diversified_recs
    
    def recommend_items_enhanced_hybrid(self,
                                       age: int,
                                       gender: str,
                                       income: float,
                                       interaction_history: List[int] = None,
                                       k: int = 10,
                                       collaborative_weight: float = 0.7,
                                       category_boost: float = 1.5,
                                       enable_category_boost: bool = True,
                                       enable_diversity: bool = True,
                                       max_per_category: int = 3) -> List[Tuple[int, float, Dict]]:
        """Generate enhanced hybrid recommendations with category awareness."""
        
        # Start with base hybrid recommendations (get more than needed)
        base_k = k * 3  # Get 3x more candidates for filtering
        
        base_recommendations = self.recommend_items_hybrid(
            age=age,
            gender=gender,
            income=income,
            interaction_history=interaction_history,
            k=base_k,
            collaborative_weight=collaborative_weight
        )
        
        if not base_recommendations:
            return []
        
        # Analyze user's category preferences
        if enable_category_boost and interaction_history:
            user_category_preferences = self._analyze_user_category_preferences(interaction_history)
            
            # Apply category-based boosting
            base_recommendations = self._boost_category_aligned_recommendations(
                base_recommendations,
                user_category_preferences,
                boost_factor=category_boost
            )
        
        # Apply diversity filtering if enabled
        if enable_diversity:
            base_recommendations = self._diversify_recommendations(
                base_recommendations,
                max_per_category=max_per_category
            )
        
        # Return top k
        return base_recommendations[:k]
    
    def recommend_items_category_focused(self,
                                        age: int,
                                        gender: str,
                                        income: float,
                                        interaction_history: List[int] = None,
                                        k: int = 10,
                                        focus_percentage: float = 0.7) -> List[Tuple[int, float, Dict]]:
        """Generate recommendations focused on user's preferred categories."""
        
        if not interaction_history:
            # Fall back to regular hybrid for users without history
            return self.recommend_items_hybrid(age, gender, income, interaction_history, k)
        
        # Analyze user preferences
        user_category_preferences = self._analyze_user_category_preferences(interaction_history)
        
        if not user_category_preferences:
            return self.recommend_items_hybrid(age, gender, income, interaction_history, k)
        
        # Get top categories (sorted by preference)
        top_categories = sorted(user_category_preferences.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Determine how many recs to focus on preferred categories
        focused_k = int(k * focus_percentage)
        exploration_k = k - focused_k
        
        # Get base recommendations
        all_recommendations = self.recommend_items_hybrid(
            age, gender, income, interaction_history, k * 2
        )
        
        # Split into focused and exploration recommendations
        focused_recs = []
        exploration_recs = []
        
        # Get user's top 3 categories
        preferred_categories = set([cat for cat, _ in top_categories[:3]])
        
        for item_id, score, item_info in all_recommendations:
            item_category = item_info.get('category_code', 'Unknown')
            
            if (item_category in preferred_categories and 
                len(focused_recs) < focused_k):
                focused_recs.append((item_id, score, item_info))
            elif len(exploration_recs) < exploration_k:
                exploration_recs.append((item_id, score, item_info))
        
        # Combine focused and exploration recommendations
        final_recommendations = focused_recs + exploration_recs
        
        return final_recommendations[:k]
    
    def get_recommendation_explanation(self,
                                     recommendations: List[Tuple[int, float, Dict]],
                                     interaction_history: List[int] = None) -> Dict:
        """Provide explanation for why these recommendations were generated."""
        
        if not recommendations:
            return {"message": "No recommendations generated"}
        
        # Analyze recommendation categories
        rec_categories = Counter()
        for _, _, item_info in recommendations:
            category = item_info.get('category_code', 'Unknown')
            rec_categories[category] += 1
        
        explanation = {
            "total_recommendations": len(recommendations),
            "categories_covered": len(rec_categories),
            "category_breakdown": dict(rec_categories.most_common())
        }
        
        # Add user preference analysis if history available
        if interaction_history:
            user_preferences = self._analyze_user_category_preferences(interaction_history)
            
            # Calculate alignment
            user_cats = set(user_preferences.keys())
            rec_cats = set(rec_categories.keys())
            alignment = len(user_cats & rec_cats) / len(rec_cats) * 100 if rec_cats else 0
            
            explanation.update({
                "user_category_preferences": user_preferences,
                "alignment_percentage": round(alignment, 1),
                "matched_categories": list(user_cats & rec_cats),
                "new_categories": list(rec_cats - user_cats)
            })
        
        return explanation


def demo_enhanced_recommendations():
    """Demo the enhanced recommendation engine."""
    
    print("🚀 ENHANCED RECOMMENDATION ENGINE DEMO")
    print("="*70)
    
    # Initialize enhanced engine
    engine = EnhancedRecommendationEngine()
    
    # Get a real user for testing
    real_user_selector = RealUserSelector()
    test_users = real_user_selector.get_real_users(n=3, min_interactions=15)
    
    for user in test_users:
        print(f"\n📊 Testing User {user['user_id']} ({user['age']}yr {user['gender']}):")
        print(f"   Interaction History: {len(user['interaction_history'])} items")
        
        # Test different recommendation methods
        methods = [
            ("Original Hybrid", lambda: engine.recommend_items_hybrid(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:20],
                k=10,
                collaborative_weight=0.7
            )),
            ("Enhanced Hybrid", lambda: engine.recommend_items_enhanced_hybrid(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:20],
                k=10,
                collaborative_weight=0.7,
                category_boost=1.5
            )),
            ("Category Focused", lambda: engine.recommend_items_category_focused(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:20],
                k=10,
                focus_percentage=0.8
            ))
        ]
        
        for method_name, method_func in methods:
            try:
                recs = method_func()
                explanation = engine.get_recommendation_explanation(
                    recs, user['interaction_history'][:20]
                )
                
                print(f"\n   🎯 {method_name}:")
                print(f"      Categories: {explanation.get('category_breakdown', {})}")
                print(f"      Alignment: {explanation.get('alignment_percentage', 'N/A')}%")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:40]}...")
    
    print(f"\n✅ Enhanced recommendation engine demo completed!")


if __name__ == "__main__":
    demo_enhanced_recommendations()