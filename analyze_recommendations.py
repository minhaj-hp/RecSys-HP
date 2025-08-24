#!/usr/bin/env python3
"""
Recommendation Analysis Script

This script compares recommendations from both training approaches:
1. 2-phase training (pre-trained item tower + joint fine-tuning)
2. Single joint training (end-to-end optimization)

It analyzes:
- Category alignment between user interactions and recommendations
- Diversity of recommended categories
- Overlap between the two approaches
- Performance on real users

Usage:
    python analyze_recommendations.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector


class RecommendationAnalyzer:
    """Analyzer for comparing different recommendation approaches."""
    
    def __init__(self):
        self.recommendation_engine = None
        self.real_user_selector = None
        self.items_df = None
        self.setup_engines()
    
    def setup_engines(self):
        """Setup recommendation engines and data."""
        print("Loading recommendation engines...")
        
        try:
            # Load recommendation engine (assumes trained model artifacts exist)
            self.recommendation_engine = RecommendationEngine()
            print("✅ Recommendation engine loaded")
        except Exception as e:
            print(f"❌ Error loading recommendation engine: {e}")
            return
        
        try:
            # Load real user selector
            self.real_user_selector = RealUserSelector()
            print("✅ Real user selector loaded")
        except Exception as e:
            print(f"❌ Error loading real user selector: {e}")
        
        # Load items data for category analysis
        self.items_df = pd.read_csv("datasets/items.csv")
        print(f"✅ Loaded {len(self.items_df)} items")
    
    def get_item_categories(self, item_ids: List[int]) -> List[str]:
        """Get category codes for given item IDs."""
        categories = []
        for item_id in item_ids:
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                categories.append(item_row.iloc[0]['category_code'])
            else:
                categories.append('unknown')
        return categories
    
    def analyze_user_recommendations(self, 
                                   user_profile: Dict,
                                   recommendation_types: List[str] = None) -> Dict:
        """Analyze recommendations for a single user across different approaches."""
        
        if recommendation_types is None:
            recommendation_types = ['collaborative', 'hybrid', 'content']
        
        results = {
            'user_profile': user_profile,
            'interaction_categories': [],
            'recommendations': {},
            'category_analysis': {}
        }
        
        # Get categories from user's interaction history
        if user_profile['interaction_history']:
            results['interaction_categories'] = self.get_item_categories(
                user_profile['interaction_history']
            )
        
        # Get recommendations for each type
        for rec_type in recommendation_types:
            try:
                if rec_type == 'collaborative':
                    recs = self.recommendation_engine.recommend_items_collaborative(
                        age=user_profile['age'],
                        gender=user_profile['gender'],
                        income=user_profile['income'],
                        interaction_history=user_profile['interaction_history'],
                        k=10
                    )
                elif rec_type == 'hybrid':
                    recs = self.recommendation_engine.recommend_items_hybrid(
                        age=user_profile['age'],
                        gender=user_profile['gender'],
                        income=user_profile['income'],
                        interaction_history=user_profile['interaction_history'],
                        k=10
                    )
                elif rec_type == 'content' and user_profile['interaction_history']:
                    recs = self.recommendation_engine.recommend_items_content_based(
                        seed_item_id=user_profile['interaction_history'][-1],
                        k=10
                    )
                else:
                    continue
                
                # Extract item IDs and categories
                item_ids = [item_id for item_id, score, info in recs]
                rec_categories = self.get_item_categories(item_ids)
                
                results['recommendations'][rec_type] = {
                    'items': recs,
                    'item_ids': item_ids,
                    'categories': rec_categories,
                    'scores': [score for item_id, score, info in recs]
                }
                
                # Analyze category alignment
                results['category_analysis'][rec_type] = self.analyze_category_alignment(
                    results['interaction_categories'],
                    rec_categories
                )
                
            except Exception as e:
                print(f"Error generating {rec_type} recommendations: {e}")
        
        return results
    
    def analyze_category_alignment(self, 
                                 interaction_categories: List[str], 
                                 recommendation_categories: List[str]) -> Dict:
        """Analyze alignment between interaction and recommendation categories."""
        
        if not interaction_categories:
            return {
                'overlap_ratio': 0.0,
                'unique_interaction_categories': 0,
                'unique_recommendation_categories': len(set(recommendation_categories)),
                'common_categories': [],
                'category_distribution': Counter(recommendation_categories)
            }
        
        interaction_set = set(interaction_categories)
        recommendation_set = set(recommendation_categories)
        
        common_categories = interaction_set.intersection(recommendation_set)
        overlap_ratio = len(common_categories) / len(interaction_set) if interaction_set else 0.0
        
        return {
            'overlap_ratio': overlap_ratio,
            'unique_interaction_categories': len(interaction_set),
            'unique_recommendation_categories': len(recommendation_set),
            'common_categories': list(common_categories),
            'category_distribution': Counter(recommendation_categories),
            'interaction_category_distribution': Counter(interaction_categories)
        }
    
    def compare_recommendation_approaches(self, 
                                        users_sample: List[Dict],
                                        approaches: List[str] = None) -> Dict:
        """Compare different recommendation approaches across multiple users."""
        
        if approaches is None:
            approaches = ['collaborative', 'hybrid', 'content']
        
        comparison_results = {
            'approach_stats': defaultdict(list),
            'cross_approach_analysis': {},
            'user_results': []
        }
        
        print(f"Analyzing {len(users_sample)} users across {len(approaches)} approaches...")
        
        for i, user in enumerate(users_sample):
            print(f"Analyzing user {i+1}/{len(users_sample)}...")
            
            user_results = self.analyze_user_recommendations(user, approaches)
            comparison_results['user_results'].append(user_results)
            
            # Aggregate stats by approach
            for approach in approaches:
                if approach in user_results['category_analysis']:
                    analysis = user_results['category_analysis'][approach]
                    comparison_results['approach_stats'][approach].append({
                        'overlap_ratio': analysis['overlap_ratio'],
                        'unique_rec_categories': analysis['unique_recommendation_categories'],
                        'common_categories_count': len(analysis['common_categories'])
                    })
        
        # Calculate aggregate statistics
        for approach in approaches:
            stats = comparison_results['approach_stats'][approach]
            if stats:
                comparison_results['approach_stats'][approach] = {
                    'avg_overlap_ratio': np.mean([s['overlap_ratio'] for s in stats]),
                    'std_overlap_ratio': np.std([s['overlap_ratio'] for s in stats]),
                    'avg_unique_categories': np.mean([s['unique_rec_categories'] for s in stats]),
                    'avg_common_categories': np.mean([s['common_categories_count'] for s in stats]),
                    'total_users': len(stats)
                }
        
        # Cross-approach analysis
        comparison_results['cross_approach_analysis'] = self.cross_approach_analysis(
            comparison_results['user_results'], approaches
        )
        
        return comparison_results
    
    def cross_approach_analysis(self, user_results: List[Dict], approaches: List[str]) -> Dict:
        """Analyze similarities and differences between approaches."""
        
        cross_analysis = {
            'item_overlap': defaultdict(dict),
            'category_overlap': defaultdict(dict),
            'score_correlation': defaultdict(dict)
        }
        
        for user_result in user_results:
            recommendations = user_result['recommendations']
            
            # Compare each pair of approaches
            for i, approach1 in enumerate(approaches):
                for approach2 in approaches[i+1:]:
                    if approach1 in recommendations and approach2 in recommendations:
                        
                        # Item overlap
                        items1 = set(recommendations[approach1]['item_ids'])
                        items2 = set(recommendations[approach2]['item_ids'])
                        item_overlap_ratio = len(items1.intersection(items2)) / len(items1.union(items2))
                        
                        # Category overlap
                        cats1 = set(recommendations[approach1]['categories'])
                        cats2 = set(recommendations[approach2]['categories'])
                        cat_overlap_ratio = len(cats1.intersection(cats2)) / len(cats1.union(cats2)) if cats1.union(cats2) else 0
                        
                        # Store results
                        pair_key = f"{approach1}_vs_{approach2}"
                        if pair_key not in cross_analysis['item_overlap']:
                            cross_analysis['item_overlap'][pair_key] = []
                            cross_analysis['category_overlap'][pair_key] = []
                        
                        cross_analysis['item_overlap'][pair_key].append(item_overlap_ratio)
                        cross_analysis['category_overlap'][pair_key].append(cat_overlap_ratio)
        
        # Calculate averages
        for pair_key in cross_analysis['item_overlap']:
            cross_analysis['item_overlap'][pair_key] = {
                'avg': np.mean(cross_analysis['item_overlap'][pair_key]),
                'std': np.std(cross_analysis['item_overlap'][pair_key])
            }
            cross_analysis['category_overlap'][pair_key] = {
                'avg': np.mean(cross_analysis['category_overlap'][pair_key]),
                'std': np.std(cross_analysis['category_overlap'][pair_key])
            }
        
        return cross_analysis
    
    def generate_report(self, comparison_results: Dict, output_file: str = "recommendation_analysis_report.md"):
        """Generate a comprehensive analysis report."""
        
        report = []
        report.append("# Recommendation System Analysis Report")
        report.append(f"Generated: {pd.Timestamp.now()}")
        report.append("")
        
        # Overall Statistics
        report.append("## Overall Statistics")
        report.append("")
        
        for approach, stats in comparison_results['approach_stats'].items():
            if isinstance(stats, dict):
                report.append(f"### {approach.title()} Recommendations")
                report.append(f"- **Average Category Overlap**: {stats['avg_overlap_ratio']:.3f} ± {stats['std_overlap_ratio']:.3f}")
                report.append(f"- **Average Unique Categories per User**: {stats['avg_unique_categories']:.1f}")
                report.append(f"- **Average Common Categories**: {stats['avg_common_categories']:.1f}")
                report.append(f"- **Users Analyzed**: {stats['total_users']}")
                report.append("")
        
        # Cross-Approach Analysis
        report.append("## Cross-Approach Comparison")
        report.append("")
        
        cross_analysis = comparison_results['cross_approach_analysis']
        
        report.append("### Item Overlap Between Approaches")
        for pair, overlap_stats in cross_analysis['item_overlap'].items():
            report.append(f"- **{pair.replace('_', ' ').title()}**: {overlap_stats['avg']:.3f} ± {overlap_stats['std']:.3f}")
        report.append("")
        
        report.append("### Category Overlap Between Approaches")
        for pair, overlap_stats in cross_analysis['category_overlap'].items():
            report.append(f"- **{pair.replace('_', ' ').title()}**: {overlap_stats['avg']:.3f} ± {overlap_stats['std']:.3f}")
        report.append("")
        
        # Category Alignment Analysis
        report.append("## Category Alignment Analysis")
        report.append("")
        report.append("Category alignment measures how well recommendations match the categories")
        report.append("of items users have previously interacted with.")
        report.append("")
        
        # Find best performing approach
        best_approach = max(
            comparison_results['approach_stats'].keys(),
            key=lambda k: comparison_results['approach_stats'][k]['avg_overlap_ratio'] 
            if isinstance(comparison_results['approach_stats'][k], dict) else 0
        )
        
        report.append(f"**Best Category Alignment**: {best_approach.title()} approach")
        report.append("")
        
        # Recommendations
        report.append("## Key Findings & Recommendations")
        report.append("")
        
        # Analyze overlap ratios to provide insights
        overlap_ratios = {
            k: v['avg_overlap_ratio'] for k, v in comparison_results['approach_stats'].items()
            if isinstance(v, dict)
        }
        
        if overlap_ratios:
            avg_overlap = np.mean(list(overlap_ratios.values()))
            if avg_overlap > 0.5:
                report.append("✅ **Strong Category Alignment**: Recommendations show good alignment with user interaction patterns.")
            elif avg_overlap > 0.3:
                report.append("⚠️ **Moderate Category Alignment**: Some alignment present but room for improvement.")
            else:
                report.append("❌ **Weak Category Alignment**: Recommendations may be too diverse or not well-aligned with user preferences.")
            
            report.append("")
            
            # Compare approaches
            if len(overlap_ratios) > 1:
                sorted_approaches = sorted(overlap_ratios.items(), key=lambda x: x[1], reverse=True)
                report.append("### Approach Rankings (by category alignment):")
                for i, (approach, ratio) in enumerate(sorted_approaches, 1):
                    report.append(f"{i}. **{approach.title()}**: {ratio:.3f}")
                report.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Analysis report saved to: {output_file}")
        return '\n'.join(report)
    
    def visualize_results(self, comparison_results: Dict, save_plots: bool = True):
        """Create visualizations for the analysis results."""
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recommendation System Analysis', fontsize=16, fontweight='bold')
        
        # 1. Category Overlap by Approach
        ax1 = axes[0, 0]
        approaches = []
        overlap_means = []
        overlap_stds = []
        
        for approach, stats in comparison_results['approach_stats'].items():
            if isinstance(stats, dict):
                approaches.append(approach.title())
                overlap_means.append(stats['avg_overlap_ratio'])
                overlap_stds.append(stats['std_overlap_ratio'])
        
        bars1 = ax1.bar(approaches, overlap_means, yerr=overlap_stds, capsize=5, alpha=0.7)
        ax1.set_title('Average Category Overlap by Approach')
        ax1.set_ylabel('Category Overlap Ratio')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, overlap_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Cross-Approach Item Overlap
        ax2 = axes[0, 1]
        cross_analysis = comparison_results['cross_approach_analysis']
        
        pair_names = []
        item_overlaps = []
        
        for pair, overlap_stats in cross_analysis['item_overlap'].items():
            pair_names.append(pair.replace('_vs_', ' vs ').title())
            item_overlaps.append(overlap_stats['avg'])
        
        if pair_names:
            bars2 = ax2.bar(pair_names, item_overlaps, alpha=0.7, color='coral')
            ax2.set_title('Item Overlap Between Approaches')
            ax2.set_ylabel('Item Overlap Ratio')
            ax2.set_ylim(0, 1)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, overlap in zip(bars2, item_overlaps):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{overlap:.3f}', ha='center', va='bottom')
        
        # 3. Category Diversity
        ax3 = axes[1, 0]
        unique_categories = []
        for approach, stats in comparison_results['approach_stats'].items():
            if isinstance(stats, dict):
                unique_categories.append(stats['avg_unique_categories'])
        
        bars3 = ax3.bar(approaches, unique_categories, alpha=0.7, color='lightgreen')
        ax3.set_title('Average Unique Categories per Recommendation')
        ax3.set_ylabel('Number of Unique Categories')
        
        for bar, cats in zip(bars3, unique_categories):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{cats:.1f}', ha='center', va='bottom')
        
        # 4. Category vs Item Overlap Comparison
        ax4 = axes[1, 1]
        
        if cross_analysis['item_overlap'] and cross_analysis['category_overlap']:
            pairs = list(cross_analysis['item_overlap'].keys())
            item_overlaps = [cross_analysis['item_overlap'][p]['avg'] for p in pairs]
            cat_overlaps = [cross_analysis['category_overlap'][p]['avg'] for p in pairs]
            
            x = np.arange(len(pairs))
            width = 0.35
            
            bars4a = ax4.bar(x - width/2, item_overlaps, width, label='Item Overlap', alpha=0.7)
            bars4b = ax4.bar(x + width/2, cat_overlaps, width, label='Category Overlap', alpha=0.7)
            
            ax4.set_title('Item vs Category Overlap Between Approaches')
            ax4.set_ylabel('Overlap Ratio')
            ax4.set_xticks(x)
            ax4.set_xticklabels([p.replace('_vs_', ' vs ') for p in pairs], rotation=45, ha='right')
            ax4.legend()
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('recommendation_analysis_plots.png', dpi=300, bbox_inches='tight')
            print("✅ Plots saved to: recommendation_analysis_plots.png")
        
        plt.show()


def main():
    """Main function to run the recommendation analysis."""
    
    print("🔍 Starting Recommendation Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RecommendationAnalyzer()
    
    if analyzer.recommendation_engine is None:
        print("❌ Cannot proceed without recommendation engine. Please ensure model is trained.")
        return
    
    # Get sample of real users for analysis
    print("Getting real user sample...")
    try:
        real_users = analyzer.real_user_selector.get_real_users(n=20, min_interactions=3)
        print(f"✅ Loaded {len(real_users)} real users for analysis")
    except Exception as e:
        print(f"❌ Error loading real users: {e}")
        # Fallback to synthetic users
        real_users = [
            {
                'age': 32, 'gender': 'male', 'income': 75000,
                'interaction_history': [1000978, 1001588, 1001618, 1002039]
            },
            {
                'age': 28, 'gender': 'female', 'income': 45000,
                'interaction_history': [1003456, 1004567, 1005678]
            },
            {
                'age': 45, 'gender': 'male', 'income': 85000,
                'interaction_history': [1006789, 1007890, 1008901, 1009012, 1010123]
            }
        ]
        print(f"Using {len(real_users)} synthetic users for analysis")
    
    # Run comprehensive analysis
    print("Running recommendation analysis...")
    approaches = ['collaborative', 'hybrid', 'content']
    
    comparison_results = analyzer.compare_recommendation_approaches(
        users_sample=real_users,
        approaches=approaches
    )
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_report(comparison_results)
    
    # Create visualizations
    print("Creating visualizations...")
    try:
        analyzer.visualize_results(comparison_results, save_plots=True)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    for approach, stats in comparison_results['approach_stats'].items():
        if isinstance(stats, dict):
            print(f"{approach.title()}: {stats['avg_overlap_ratio']:.3f} avg category overlap")
    
    print(f"\n✅ Analysis complete! Check:")
    print("   📄 recommendation_analysis_report.md")
    print("   📊 recommendation_analysis_plots.png")


if __name__ == "__main__":
    main()