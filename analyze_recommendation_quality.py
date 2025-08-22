#!/usr/bin/env python3
"""
Comprehensive analysis of recommendation quality from the two-tower model.
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import time

sys.path.append('/home/user/Desktop/RecSys-HP')
from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector

def analyze_score_distribution():
    """Analyze the distribution of recommendation scores."""
    
    print("📊 SCORE DISTRIBUTION ANALYSIS")
    print("="*50)
    
    try:
        engine = RecommendationEngine()
        real_user_selector = RealUserSelector()
        
        # Get multiple users for comprehensive analysis
        test_users = real_user_selector.get_real_users(n=10, min_interactions=10)
        
        all_scores = {
            'collaborative': [],
            'hybrid': [],
            'content': []
        }
        
        print(f"Testing with {len(test_users)} users...")
        
        for i, user in enumerate(test_users):
            print(f"\nUser {i+1}/10 - {user['user_id']} ({user['age']}yr {user['gender']}):")
            
            # Test collaborative filtering
            try:
                collab_recs = engine.recommend_items_collaborative(
                    age=user['age'],
                    gender=user['gender'],
                    income=user['income'],
                    interaction_history=user['interaction_history'][:20],
                    k=20
                )
                collab_scores = [score for _, score, _ in collab_recs]
                all_scores['collaborative'].extend(collab_scores)
                
                print(f"   Collaborative: {min(collab_scores):.4f} - {max(collab_scores):.4f} (std: {np.std(collab_scores):.4f})")
                
            except Exception as e:
                print(f"   Collaborative failed: {e}")
            
            # Test hybrid
            try:
                hybrid_recs = engine.recommend_items_hybrid(
                    age=user['age'],
                    gender=user['gender'],
                    income=user['income'],
                    interaction_history=user['interaction_history'][:20],
                    k=20,
                    collaborative_weight=0.7
                )
                hybrid_scores = [score for _, score, _ in hybrid_recs]
                all_scores['hybrid'].extend(hybrid_scores)
                
                print(f"   Hybrid: {min(hybrid_scores):.4f} - {max(hybrid_scores):.4f} (std: {np.std(hybrid_scores):.4f})")
                
            except Exception as e:
                print(f"   Hybrid failed: {e}")
            
            # Test content-based (if user has history)
            if user['interaction_history']:
                try:
                    content_recs = engine.recommend_items_content_based(
                        seed_item_id=user['interaction_history'][0],
                        k=20
                    )
                    content_scores = [score for _, score, _ in content_recs]
                    all_scores['content'].extend(content_scores)
                    
                    print(f"   Content: {min(content_scores):.4f} - {max(content_scores):.4f} (std: {np.std(content_scores):.4f})")
                    
                except Exception as e:
                    print(f"   Content failed: {e}")
        
        # Overall score analysis
        print(f"\n📈 OVERALL SCORE STATISTICS:")
        for method, scores in all_scores.items():
            if scores:
                print(f"\n{method.upper()}:")
                print(f"   Total scores: {len(scores)}")
                print(f"   Range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"   Mean: {np.mean(scores):.4f}")
                print(f"   Std: {np.std(scores):.4f}")
                print(f"   Variance: {np.var(scores):.6f}")
                
                # Score distribution percentiles
                percentiles = [10, 25, 50, 75, 90]
                perc_values = np.percentile(scores, percentiles)
                print(f"   Percentiles: {dict(zip(percentiles, perc_values))}")
                
                # Quality assessment
                score_range = max(scores) - min(scores)
                if score_range < 0.1:
                    print(f"   ⚠️  WARNING: Low score range ({score_range:.4f}) - poor discrimination")
                elif score_range < 0.3:
                    print(f"   ⚠️  CAUTION: Moderate score range ({score_range:.4f})")
                else:
                    print(f"   ✅ GOOD: Wide score range ({score_range:.4f})")
                
                if np.var(scores) < 0.001:
                    print(f"   ⚠️  WARNING: Very low variance - poor ranking ability")
                elif np.var(scores) < 0.01:
                    print(f"   ⚠️  CAUTION: Low variance")
                else:
                    print(f"   ✅ GOOD: Adequate variance for ranking")
        
        return all_scores
        
    except Exception as e:
        print(f"❌ Score analysis failed: {e}")
        return None

def analyze_category_alignment():
    """Analyze how well recommendations align with user category preferences."""
    
    print(f"\n🎯 CATEGORY ALIGNMENT ANALYSIS")
    print("="*40)
    
    try:
        engine = RecommendationEngine()
        real_user_selector = RealUserSelector()
        
        test_users = real_user_selector.get_real_users(n=5, min_interactions=15)
        
        alignment_results = []
        
        for user in test_users:
            print(f"\nUser {user['user_id']} ({user['age']}yr {user['gender']}):")
            
            # Get user's detailed interactions
            user_details = real_user_selector.get_user_interaction_details(user['user_id'])
            
            # Analyze user's category preferences
            user_categories = []
            for interaction in user_details['timeline']:
                category = interaction.get('category_code', 'Unknown')
                user_categories.append(category)
            
            user_category_counts = Counter(user_categories)
            total_user_interactions = len(user_categories)
            
            print(f"   User's top categories:")
            for category, count in user_category_counts.most_common(3):
                percentage = (count / total_user_interactions) * 100
                print(f"     {category}: {count} ({percentage:.1f}%)")
            
            # Get recommendations
            recs = engine.recommend_items_hybrid(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:20],
                k=20,
                collaborative_weight=0.7
            )
            
            # Analyze recommendation categories
            rec_categories = []
            for _, _, item_info in recs:
                category = item_info.get('category_code', 'Unknown')
                rec_categories.append(category)
            
            rec_category_counts = Counter(rec_categories)
            
            print(f"   Recommendation categories:")
            for category, count in rec_category_counts.most_common(3):
                percentage = (count / len(rec_categories)) * 100
                match = "✅" if category in user_category_counts else "🆕"
                print(f"     {category}: {count} ({percentage:.1f}%) {match}")
            
            # Calculate alignment metrics
            user_cats = set(user_category_counts.keys())
            rec_cats = set(rec_category_counts.keys())
            
            intersection = user_cats & rec_cats
            alignment_percentage = len(intersection) / len(rec_cats) * 100 if rec_cats else 0
            
            # Calculate weighted alignment (by user preference strength)
            weighted_alignment = 0
            for category in intersection:
                user_weight = user_category_counts[category] / total_user_interactions
                rec_weight = rec_category_counts[category] / len(rec_categories)
                weighted_alignment += min(user_weight, rec_weight)
            
            alignment_results.append({
                'user_id': user['user_id'],
                'alignment_percentage': alignment_percentage,
                'weighted_alignment': weighted_alignment * 100,
                'user_categories': len(user_cats),
                'rec_categories': len(rec_cats),
                'matched_categories': len(intersection)
            })
            
            print(f"   Alignment: {alignment_percentage:.1f}% ({len(intersection)}/{len(rec_cats)} categories)")
            print(f"   Weighted alignment: {weighted_alignment * 100:.1f}%")
        
        # Overall alignment analysis
        print(f"\n📊 OVERALL ALIGNMENT STATISTICS:")
        avg_alignment = np.mean([r['alignment_percentage'] for r in alignment_results])
        avg_weighted = np.mean([r['weighted_alignment'] for r in alignment_results])
        avg_user_cats = np.mean([r['user_categories'] for r in alignment_results])
        avg_rec_cats = np.mean([r['rec_categories'] for r in alignment_results])
        
        print(f"   Average alignment: {avg_alignment:.1f}%")
        print(f"   Average weighted alignment: {avg_weighted:.1f}%")
        print(f"   Average user categories: {avg_user_cats:.1f}")
        print(f"   Average rec categories: {avg_rec_cats:.1f}")
        
        # Quality assessment
        if avg_alignment < 20:
            print(f"   ❌ POOR: Very low category alignment")
        elif avg_alignment < 40:
            print(f"   ⚠️  FAIR: Low category alignment")
        elif avg_alignment < 60:
            print(f"   ✅ GOOD: Moderate category alignment")
        else:
            print(f"   🎉 EXCELLENT: High category alignment")
        
        return alignment_results
        
    except Exception as e:
        print(f"❌ Category alignment analysis failed: {e}")
        return None

def analyze_diversity_metrics():
    """Analyze diversity metrics in recommendations."""
    
    print(f"\n🌈 DIVERSITY ANALYSIS")
    print("="*30)
    
    try:
        engine = RecommendationEngine()
        real_user_selector = RealUserSelector()
        
        test_users = real_user_selector.get_real_users(n=5, min_interactions=10)
        
        diversity_results = []
        
        for user in test_users:
            print(f"\nUser {user['user_id']}:")
            
            # Get recommendations
            recs = engine.recommend_items_hybrid(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:20],
                k=20,
                collaborative_weight=0.7
            )
            
            # Extract features for diversity analysis
            categories = [item_info.get('category_code', 'Unknown') for _, _, item_info in recs]
            brands = [item_info.get('brand', 'Unknown') for _, _, item_info in recs]
            prices = [item_info.get('price', 0) for _, _, item_info in recs]
            
            # Calculate diversity metrics
            category_diversity = len(set(categories)) / len(categories) if categories else 0
            brand_diversity = len(set(brands)) / len(brands) if brands else 0
            
            # Price diversity (coefficient of variation)
            price_diversity = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Intra-list diversity (average pairwise dissimilarity)
            category_counts = Counter(categories)
            gini_categories = 1 - sum((count / len(categories)) ** 2 for count in category_counts.values())
            
            diversity_results.append({
                'user_id': user['user_id'],
                'category_diversity': category_diversity,
                'brand_diversity': brand_diversity,
                'price_diversity': price_diversity,
                'gini_categories': gini_categories,
                'unique_categories': len(set(categories)),
                'unique_brands': len(set(brands))
            })
            
            print(f"   Categories: {len(set(categories))} unique ({category_diversity:.2f} ratio)")
            print(f"   Brands: {len(set(brands))} unique ({brand_diversity:.2f} ratio)")
            print(f"   Price range: ${min(prices):.2f} - ${max(prices):.2f}")
            print(f"   Gini (categories): {gini_categories:.2f}")
        
        # Overall diversity statistics
        print(f"\n📊 OVERALL DIVERSITY STATISTICS:")
        avg_cat_diversity = np.mean([r['category_diversity'] for r in diversity_results])
        avg_brand_diversity = np.mean([r['brand_diversity'] for r in diversity_results])
        avg_gini = np.mean([r['gini_categories'] for r in diversity_results])
        avg_unique_cats = np.mean([r['unique_categories'] for r in diversity_results])
        
        print(f"   Average category diversity: {avg_cat_diversity:.2f}")
        print(f"   Average brand diversity: {avg_brand_diversity:.2f}")
        print(f"   Average Gini coefficient: {avg_gini:.2f}")
        print(f"   Average unique categories: {avg_unique_cats:.1f}")
        
        # Quality assessment
        if avg_cat_diversity < 0.3:
            print(f"   ❌ POOR: Low category diversity - recommendations too similar")
        elif avg_cat_diversity < 0.5:
            print(f"   ⚠️  FAIR: Moderate category diversity")
        else:
            print(f"   ✅ GOOD: High category diversity")
        
        return diversity_results
        
    except Exception as e:
        print(f"❌ Diversity analysis failed: {e}")
        return None

def analyze_embedding_quality():
    """Analyze the quality of user and item embeddings."""
    
    print(f"\n🧠 EMBEDDING QUALITY ANALYSIS")
    print("="*35)
    
    try:
        engine = RecommendationEngine()
        real_user_selector = RealUserSelector()
        
        test_users = real_user_selector.get_real_users(n=3, min_interactions=10)
        
        user_embeddings = []
        user_item_similarities = []
        
        for user in test_users:
            print(f"\nUser {user['user_id']}:")
            
            # Get user embedding
            user_emb = engine.get_user_embedding(
                age=user['age'],
                gender=user['gender'],
                income=user['income'],
                interaction_history=user['interaction_history'][:10]
            )
            
            user_embeddings.append(user_emb)
            
            print(f"   User embedding shape: {user_emb.shape}")
            print(f"   User embedding norm: {np.linalg.norm(user_emb):.4f}")
            print(f"   User embedding mean: {user_emb.mean():.4f}")
            print(f"   User embedding std: {user_emb.std():.4f}")
            
            # Get embeddings for user's interaction history
            item_similarities = []
            for item_id in user['interaction_history'][:5]:
                item_emb = engine.get_item_embedding(item_id)
                if item_emb is not None:
                    similarity = np.dot(user_emb, item_emb)
                    item_similarities.append(similarity)
            
            if item_similarities:
                user_item_similarities.extend(item_similarities)
                print(f"   Avg similarity with interacted items: {np.mean(item_similarities):.4f}")
                print(f"   Similarity range: {min(item_similarities):.4f} - {max(item_similarities):.4f}")
        
        # Analyze user embedding diversity
        if len(user_embeddings) > 1:
            user_embeddings = np.array(user_embeddings)
            
            # User-user similarities
            user_similarities = []
            for i in range(len(user_embeddings)):
                for j in range(i+1, len(user_embeddings)):
                    sim = np.dot(user_embeddings[i], user_embeddings[j])
                    user_similarities.append(sim)
            
            print(f"\n📊 USER EMBEDDING ANALYSIS:")
            print(f"   User-user similarities: {np.mean(user_similarities):.4f} ± {np.std(user_similarities):.4f}")
            print(f"   User-item similarities: {np.mean(user_item_similarities):.4f} ± {np.std(user_item_similarities):.4f}")
            
            # Quality assessment
            if np.mean(user_similarities) > 0.9:
                print(f"   ⚠️  WARNING: Users too similar - possible embedding collapse")
            elif np.mean(user_similarities) > 0.7:
                print(f"   ⚠️  CAUTION: High user similarity - limited personalization")
            else:
                print(f"   ✅ GOOD: Adequate user embedding diversity")
        
        return {
            'user_embeddings': user_embeddings,
            'user_similarities': user_similarities if len(user_embeddings) > 1 else [],
            'user_item_similarities': user_item_similarities
        }
        
    except Exception as e:
        print(f"❌ Embedding analysis failed: {e}")
        return None

def analyze_performance_metrics():
    """Analyze performance and efficiency metrics."""
    
    print(f"\n⚡ PERFORMANCE ANALYSIS")
    print("="*25)
    
    try:
        engine = RecommendationEngine()
        real_user_selector = RealUserSelector()
        
        test_user = real_user_selector.get_real_users(n=1, min_interactions=10)[0]
        
        # Test recommendation generation speed
        print("Testing recommendation generation speed...")
        
        methods = [
            ('Collaborative', lambda: engine.recommend_items_collaborative(
                age=test_user['age'], gender=test_user['gender'], 
                income=test_user['income'], interaction_history=test_user['interaction_history'][:20], k=10
            )),
            ('Hybrid', lambda: engine.recommend_items_hybrid(
                age=test_user['age'], gender=test_user['gender'], 
                income=test_user['income'], interaction_history=test_user['interaction_history'][:20], k=10
            )),
        ]
        
        for method_name, method_func in methods:
            times = []
            for _ in range(5):  # Run 5 times for average
                start_time = time.time()
                recs = method_func()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            print(f"   {method_name}: {avg_time:.3f}s ± {np.std(times):.3f}s")
            
            if avg_time > 1.0:
                print(f"     ⚠️  SLOW: Consider optimization")
            elif avg_time > 0.5:
                print(f"     ⚠️  MODERATE: Acceptable for real-time")
            else:
                print(f"     ✅ FAST: Good for real-time recommendations")
        
        # Test scalability with different recommendation counts
        print(f"\nTesting scalability...")
        for k in [10, 50, 100]:
            start_time = time.time()
            recs = engine.recommend_items_hybrid(
                age=test_user['age'], gender=test_user['gender'], 
                income=test_user['income'], interaction_history=test_user['interaction_history'][:20], k=k
            )
            end_time = time.time()
            
            print(f"   {k} recommendations: {end_time - start_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return False

def generate_quality_report():
    """Generate a comprehensive quality report."""
    
    print(f"\n📋 COMPREHENSIVE QUALITY REPORT")
    print("="*40)
    
    # Run all analyses
    score_results = analyze_score_distribution()
    alignment_results = analyze_category_alignment()
    diversity_results = analyze_diversity_metrics()
    embedding_results = analyze_embedding_quality()
    performance_results = analyze_performance_metrics()
    
    # Generate summary
    print(f"\n🎯 QUALITY SUMMARY:")
    
    issues = []
    strengths = []
    
    # Check score quality
    if score_results:
        for method, scores in score_results.items():
            if scores:
                score_variance = np.var(scores)
                score_range = max(scores) - min(scores)
                
                if score_variance < 0.001:
                    issues.append(f"Low {method} score variance ({score_variance:.6f})")
                if score_range < 0.1:
                    issues.append(f"Narrow {method} score range ({score_range:.4f})")
                
                if score_variance > 0.01 and score_range > 0.3:
                    strengths.append(f"Good {method} score discrimination")
    
    # Check alignment quality
    if alignment_results:
        avg_alignment = np.mean([r['alignment_percentage'] for r in alignment_results])
        if avg_alignment < 30:
            issues.append(f"Low category alignment ({avg_alignment:.1f}%)")
        elif avg_alignment > 50:
            strengths.append(f"Good category alignment ({avg_alignment:.1f}%)")
    
    # Check diversity
    if diversity_results:
        avg_diversity = np.mean([r['category_diversity'] for r in diversity_results])
        if avg_diversity < 0.3:
            issues.append(f"Low category diversity ({avg_diversity:.2f})")
        elif avg_diversity > 0.5:
            strengths.append(f"Good category diversity ({avg_diversity:.2f})")
    
    # Print results
    if issues:
        print(f"\n❌ ISSUES IDENTIFIED:")
        for issue in issues:
            print(f"   • {issue}")
    
    if strengths:
        print(f"\n✅ STRENGTHS:")
        for strength in strengths:
            print(f"   • {strength}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if any("score variance" in issue for issue in issues):
        print("   • Increase embedding dimensions or add temperature scaling")
    if any("alignment" in issue for issue in issues):
        print("   • Implement category-aware recommendation boosting")
    if any("diversity" in issue for issue in issues):
        print("   • Add diversity regularization to recommendation algorithm")
    
    if not issues:
        print("   • No major issues detected - model performing well!")

def main():
    """Main analysis function."""
    
    print("🔍 TWO-TOWER RECOMMENDATION QUALITY ANALYSIS")
    print("="*60)
    
    try:
        generate_quality_report()
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()