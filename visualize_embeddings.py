#!/usr/bin/env python3
"""
User and Item Embeddings Visualization

This script creates 2D visualizations of user and item embeddings from the
two-tower recommendation system to understand:
1. User clustering by demographics and preferences
2. Item clustering by categories and characteristics  
3. User-item similarity patterns in embedding space
4. Quality of the learned representations
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference.recommendation_engine import RecommendationEngine
    print("✅ Successfully imported RecommendationEngine")
except Exception as e:
    print(f"❌ Failed to import RecommendationEngine: {e}")
    sys.exit(1)

# Optional imports for advanced visualization
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
    print("✅ scikit-learn available for t-SNE/PCA")
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not available - using PCA approximation")

try:
    import umap
    HAS_UMAP = True
    print("✅ UMAP available for advanced dimensionality reduction")
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not available - using t-SNE/PCA only")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    print("✅ Plotly available for interactive visualizations")
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available - using matplotlib only")


class EmbeddingVisualizer:
    """Visualize user and item embeddings from the two-tower system."""
    
    def __init__(self):
        print("🔧 Initializing Embedding Visualizer...")
        
        try:
            self.engine = RecommendationEngine()
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load recommendation engine: {e}")
            raise
            
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_diverse_test_users(self) -> List[Dict]:
        """Create diverse test users for embedding visualization."""
        
        return [
            # Tech professionals
            {
                'name': 'YoungTechMale', 'age': 25, 'gender': 'male', 'income': 85000,
                'profession': 'Technology', 'location': 'Urban', 'education_level': "Bachelor's",
                'marital_status': 'Single', 'interaction_history': [1000978, 1001588, 1001618, 1002000],
                'group': 'Tech_Professional', 'color': 'red'
            },
            {
                'name': 'YoungTechFemale', 'age': 27, 'gender': 'female', 'income': 78000,
                'profession': 'Technology', 'location': 'Urban', 'education_level': "Master's",
                'marital_status': 'Single', 'interaction_history': [1000980, 1001590, 1001620, 1002010],
                'group': 'Tech_Professional', 'color': 'red'
            },
            
            # Healthcare professionals
            {
                'name': 'HealthcareFemale1', 'age': 35, 'gender': 'female', 'income': 68000,
                'profession': 'Healthcare', 'location': 'Suburban', 'education_level': "Master's",
                'marital_status': 'Married', 'interaction_history': [1003000, 1003100, 1003200, 1003300],
                'group': 'Healthcare_Professional', 'color': 'blue'
            },
            {
                'name': 'HealthcareMale', 'age': 42, 'gender': 'male', 'income': 72000,
                'profession': 'Healthcare', 'location': 'Urban', 'education_level': "Master's",
                'marital_status': 'Married', 'interaction_history': [1003010, 1003110, 1003210, 1003310],
                'group': 'Healthcare_Professional', 'color': 'blue'
            },
            
            # Finance professionals  
            {
                'name': 'FinanceSenior', 'age': 45, 'gender': 'female', 'income': 120000,
                'profession': 'Finance', 'location': 'Urban', 'education_level': "Master's",
                'marital_status': 'Married', 'interaction_history': [1004000, 1004100, 1004200],
                'group': 'Finance_Professional', 'color': 'green'
            },
            
            # Students/Low income
            {
                'name': 'YoungStudent', 'age': 20, 'gender': 'male', 'income': 15000,
                'profession': 'Other', 'location': 'Urban', 'education_level': "Some College",
                'marital_status': 'Single', 'interaction_history': [1005000, 1005100, 1005200],
                'group': 'Student', 'color': 'orange'
            },
            {
                'name': 'YoungStudentFemale', 'age': 21, 'gender': 'female', 'income': 12000,
                'profession': 'Other', 'location': 'Urban', 'education_level': "Some College",
                'marital_status': 'Single', 'interaction_history': [1005010, 1005110, 1005210],
                'group': 'Student', 'color': 'orange'
            },
            
            # Seniors/Retirees
            {
                'name': 'SeniorRetiree', 'age': 67, 'gender': 'female', 'income': 35000,
                'profession': 'Other', 'location': 'Rural', 'education_level': "High School",
                'marital_status': 'Widowed', 'interaction_history': [1006000, 1006100],
                'group': 'Senior', 'color': 'purple'
            },
            
            # Zero interaction users (cold start)
            {
                'name': 'ZeroTech', 'age': 30, 'gender': 'male', 'income': 75000,
                'profession': 'Technology', 'location': 'Urban', 'education_level': "Bachelor's",
                'marital_status': 'Single', 'interaction_history': [],
                'group': 'Cold_Start', 'color': 'gray'
            },
            {
                'name': 'ZeroHealthcare', 'age': 35, 'gender': 'female', 'income': 65000,
                'profession': 'Healthcare', 'location': 'Suburban', 'education_level': "Master's",
                'marital_status': 'Married', 'interaction_history': [],
                'group': 'Cold_Start', 'color': 'gray'
            },
            {
                'name': 'ZeroSenior', 'age': 60, 'gender': 'male', 'income': 40000,
                'profession': 'Other', 'location': 'Rural', 'education_level': "High School",
                'marital_status': 'Married', 'interaction_history': [],
                'group': 'Cold_Start', 'color': 'gray'
            }
        ]
    
    def extract_user_embeddings(self, test_users: List[Dict]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract user embeddings using the UserTower."""
        
        print(f"\n📊 Extracting user embeddings...")
        
        user_embeddings = []
        user_names = []
        user_groups = []
        
        for user in test_users:
            try:
                # Get user embedding via UserTower
                embedding = self.engine.get_user_embedding_enhanced(
                    age=user['age'],
                    gender=user['gender'], 
                    income=user['income'],
                    profession=user['profession'],
                    location=user['location'],
                    education_level=user['education_level'],
                    marital_status=user['marital_status'],
                    interaction_history=user['interaction_history']
                )
                
                if embedding is not None:
                    user_embeddings.append(embedding)
                    user_names.append(user['name'])
                    user_groups.append(user['group'])
                    print(f"   ✅ {user['name']}: {embedding.shape} embedding")
                else:
                    print(f"   ❌ {user['name']}: Failed to get embedding")
                    
            except Exception as e:
                print(f"   ❌ {user['name']}: Error - {e}")
        
        if user_embeddings:
            user_embeddings = np.array(user_embeddings)
            print(f"📈 Extracted {len(user_embeddings)} user embeddings: {user_embeddings.shape}")
        else:
            print(f"❌ No user embeddings extracted!")
            
        return user_embeddings, user_names, user_groups
    
    def extract_item_embeddings(self, max_items: int = 1000) -> Tuple[np.ndarray, List[int], List[str]]:
        """Extract sample of item embeddings from FAISS index."""
        
        print(f"\n📊 Extracting item embeddings (max {max_items})...")
        
        # Get sample of items with diverse categories
        items_df = self.engine.items_df.copy()
        
        # Sample items stratified by category for diversity
        item_embeddings = []
        item_ids = []
        item_categories = []
        
        # Group by top-level category and sample
        items_df['top_category'] = items_df['category_code'].str.split('.').str[0]
        category_groups = items_df.groupby('top_category')
        
        items_per_category = min(50, max_items // len(category_groups))
        
        for category, group in category_groups:
            if len(item_embeddings) >= max_items:
                break
                
            sample_size = min(items_per_category, len(group))
            sample_items = group.sample(n=sample_size, random_state=42)
            
            for _, item in sample_items.iterrows():
                item_id = item['product_id']
                
                # Get embedding from FAISS index
                embedding = self.engine.faiss_index.get_item_embedding(item_id)
                
                if embedding is not None:
                    item_embeddings.append(embedding)
                    item_ids.append(item_id)
                    item_categories.append(category)
                    
                    if len(item_embeddings) >= max_items:
                        break
        
        if item_embeddings:
            item_embeddings = np.array(item_embeddings)
            print(f"📈 Extracted {len(item_embeddings)} item embeddings: {item_embeddings.shape}")
            
            # Show category distribution
            category_counts = pd.Series(item_categories).value_counts()
            print(f"📊 Category distribution: {dict(category_counts.head())}")
        else:
            print(f"❌ No item embeddings extracted!")
            
        return item_embeddings, item_ids, item_categories
    
    def simple_pca_2d(self, embeddings: np.ndarray) -> np.ndarray:
        """Simple PCA implementation for 2D reduction when sklearn not available."""
        
        # Center the data
        centered = embeddings - np.mean(embeddings, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project to 2D using top 2 components
        reduced = centered @ eigenvectors[:, :2]
        
        return reduced
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
        """Reduce embeddings to 2D for visualization."""
        
        print(f"🔄 Reducing dimensions using {method.upper()}...")
        
        if method == 'pca':
            if HAS_SKLEARN:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                reduced = reducer.fit_transform(embeddings)
                print(f"   ✅ PCA explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
            else:
                reduced = self.simple_pca_2d(embeddings)
                print(f"   ✅ Simple PCA reduction completed")
                
        elif method == 'tsne' and HAS_SKLEARN:
            # Use PCA first for speed if high dimensional
            if embeddings.shape[1] > 50:
                from sklearn.decomposition import PCA
                n_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
                pca = PCA(n_components=n_components, random_state=42)
                embeddings = pca.fit_transform(embeddings)
                print(f"   📉 Pre-reduced to {n_components}D with PCA")
            
            perplexity = min(30, max(5, embeddings.shape[0] - 1))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            reduced = reducer.fit_transform(embeddings)
            print(f"   ✅ t-SNE reduction completed (perplexity={perplexity})")
            
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, embeddings.shape[0]-1))
            reduced = reducer.fit_transform(embeddings)
            print(f"   ✅ UMAP reduction completed")
            
        else:
            print(f"   ⚠️  {method.upper()} not available, falling back to PCA")
            reduced = self.simple_pca_2d(embeddings)
        
        return reduced
    
    def plot_user_embeddings(self, user_embeddings: np.ndarray, user_names: List[str], 
                           user_groups: List[str], method: str = 'tsne') -> plt.Figure:
        """Create 2D plot of user embeddings."""
        
        print(f"\n📈 Creating user embeddings plot...")
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(user_embeddings, method)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for groups
        unique_groups = list(set(user_groups))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        group_colors = dict(zip(unique_groups, colors))
        
        # Plot points by group
        for group in unique_groups:
            mask = np.array(user_groups) == group
            if np.any(mask):
                x = reduced_embeddings[mask, 0]
                y = reduced_embeddings[mask, 1]
                names = np.array(user_names)[mask]
                
                ax.scatter(x, y, c=[group_colors[group]], label=group, alpha=0.7, s=100)
                
                # Add labels
                for i, name in enumerate(names):
                    ax.annotate(name, (x[i], y[i]), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.set_title(f'User Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_item_embeddings(self, item_embeddings: np.ndarray, item_categories: List[str], 
                           method: str = 'tsne') -> plt.Figure:
        """Create 2D plot of item embeddings."""
        
        print(f"\n📈 Creating item embeddings plot...")
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(item_embeddings, method)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for categories
        unique_categories = list(set(item_categories))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        category_colors = dict(zip(unique_categories, colors))
        
        # Plot points by category
        for category in unique_categories:
            mask = np.array(item_categories) == category
            if np.any(mask):
                x = reduced_embeddings[mask, 0]
                y = reduced_embeddings[mask, 1]
                
                ax.scatter(x, y, c=[category_colors[category]], label=category, 
                          alpha=0.6, s=30)
        
        ax.set_title(f'Item Embeddings Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_combined_embedding_space(self, user_embeddings: np.ndarray, item_embeddings: np.ndarray,
                                    user_names: List[str], user_groups: List[str], 
                                    item_categories: List[str], method: str = 'tsne') -> plt.Figure:
        """Create combined plot showing users and items in same embedding space."""
        
        print(f"\n📈 Creating combined embedding space plot...")
        
        # Combine embeddings
        all_embeddings = np.vstack([user_embeddings, item_embeddings])
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(all_embeddings, method)
        
        # Split back
        n_users = len(user_embeddings)
        user_reduced = reduced_embeddings[:n_users]
        item_reduced = reduced_embeddings[n_users:]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot items first (as background)
        unique_categories = list(set(item_categories))
        item_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        category_colors = dict(zip(unique_categories, item_colors))
        
        for category in unique_categories:
            mask = np.array(item_categories) == category
            if np.any(mask):
                x = item_reduced[mask, 0]
                y = item_reduced[mask, 1]
                
                ax.scatter(x, y, c=[category_colors[category]], label=f'Items: {category}', 
                          alpha=0.3, s=20, marker='.')
        
        # Plot users on top
        unique_groups = list(set(user_groups))
        user_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        group_colors = dict(zip(unique_groups, user_colors))
        
        for group in unique_groups:
            mask = np.array(user_groups) == group
            if np.any(mask):
                x = user_reduced[mask, 0]
                y = user_reduced[mask, 1]
                names = np.array(user_names)[mask]
                
                ax.scatter(x, y, c=[group_colors[group]], label=f'Users: {group}', 
                          alpha=0.8, s=150, marker='*', edgecolors='black', linewidths=0.5)
                
                # Add user labels
                for i, name in enumerate(names):
                    ax.annotate(name, (x[i], y[i]), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax.set_title(f'Combined User-Item Embedding Space ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_embedding_quality(self, user_embeddings: np.ndarray, user_groups: List[str],
                                item_embeddings: np.ndarray, item_categories: List[str]) -> Dict:
        """Analyze the quality of learned embeddings."""
        
        print(f"\n🔍 Analyzing embedding quality...")
        
        analysis = {}
        
        # User embedding analysis
        print(f"👥 User Embedding Analysis:")
        analysis['user_stats'] = {
            'count': len(user_embeddings),
            'dimensions': user_embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(user_embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(user_embeddings, axis=1))
        }
        
        # Calculate within-group vs between-group similarities for users
        if len(user_embeddings) > 1:
            user_similarities = np.dot(user_embeddings, user_embeddings.T)
            
            within_group_sims = []
            between_group_sims = []
            
            for i in range(len(user_groups)):
                for j in range(i+1, len(user_groups)):
                    sim = user_similarities[i, j]
                    if user_groups[i] == user_groups[j]:
                        within_group_sims.append(sim)
                    else:
                        between_group_sims.append(sim)
            
            analysis['user_clustering'] = {
                'within_group_similarity': np.mean(within_group_sims) if within_group_sims else 0,
                'between_group_similarity': np.mean(between_group_sims) if between_group_sims else 0,
                'separation_score': (np.mean(within_group_sims) - np.mean(between_group_sims)) if within_group_sims and between_group_sims else 0
            }
            
            print(f"   Within-group similarity: {analysis['user_clustering']['within_group_similarity']:.3f}")
            print(f"   Between-group similarity: {analysis['user_clustering']['between_group_similarity']:.3f}")
            print(f"   Separation score: {analysis['user_clustering']['separation_score']:.3f}")
        
        # Item embedding analysis
        print(f"🛍️  Item Embedding Analysis:")
        analysis['item_stats'] = {
            'count': len(item_embeddings),
            'dimensions': item_embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(item_embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(item_embeddings, axis=1))
        }
        
        print(f"   📊 Stats: {analysis['user_stats']['count']} users, {analysis['item_stats']['count']} items")
        print(f"   📐 Dimensions: {analysis['user_stats']['dimensions']}")
        print(f"   📏 User norm: {analysis['user_stats']['mean_norm']:.3f} ± {analysis['user_stats']['std_norm']:.3f}")
        print(f"   📏 Item norm: {analysis['item_stats']['mean_norm']:.3f} ± {analysis['item_stats']['std_norm']:.3f}")
        
        return analysis
    
    def save_results(self, figures: List[plt.Figure], analysis: Dict, timestamp: str):
        """Save visualization results."""
        
        print(f"\n💾 Saving visualization results...")
        
        # Save figures
        for i, fig in enumerate(figures):
            filename = f"embedding_visualization_{i+1}_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   📊 Saved figure: {filename}")
        
        # Save analysis
        analysis_file = f"embedding_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, dict):
                    json_analysis[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                        for k, v in value.items()}
                else:
                    json_analysis[key] = value
            
            json.dump(json_analysis, f, indent=2)
        
        print(f"   📄 Saved analysis: {analysis_file}")
    
    def run_visualization(self, max_items: int = 500, methods: List[str] = ['tsne']):
        """Run complete embedding visualization pipeline."""
        
        print("🚀 Starting Embedding Visualization Pipeline")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create test users
        test_users = self.create_diverse_test_users()
        print(f"👥 Created {len(test_users)} diverse test users")
        
        # Extract embeddings
        user_embeddings, user_names, user_groups = self.extract_user_embeddings(test_users)
        item_embeddings, item_ids, item_categories = self.extract_item_embeddings(max_items)
        
        if len(user_embeddings) == 0 or len(item_embeddings) == 0:
            print("❌ Failed to extract embeddings - cannot proceed")
            return
        
        # Analyze embedding quality
        analysis = self.analyze_embedding_quality(user_embeddings, user_groups, 
                                                item_embeddings, item_categories)
        
        # Create visualizations
        figures = []
        
        for method in methods:
            print(f"\n🎨 Creating visualizations with {method.upper()}...")
            
            # User embeddings plot
            user_fig = self.plot_user_embeddings(user_embeddings, user_names, user_groups, method)
            figures.append(user_fig)
            
            # Item embeddings plot (sample for visibility)
            sample_size = min(300, len(item_embeddings))
            sample_idx = np.random.choice(len(item_embeddings), sample_size, replace=False)
            item_sample_emb = item_embeddings[sample_idx]
            item_sample_cat = [item_categories[i] for i in sample_idx]
            
            item_fig = self.plot_item_embeddings(item_sample_emb, item_sample_cat, method)
            figures.append(item_fig)
            
            # Combined plot (smaller sample for clarity)
            if len(item_embeddings) > 200:
                sample_idx = np.random.choice(len(item_embeddings), 200, replace=False)
                combined_item_emb = item_embeddings[sample_idx]
                combined_item_cat = [item_categories[i] for i in sample_idx]
            else:
                combined_item_emb = item_embeddings
                combined_item_cat = item_categories
            
            combined_fig = self.plot_combined_embedding_space(
                user_embeddings, combined_item_emb, user_names, user_groups, 
                combined_item_cat, method
            )
            figures.append(combined_fig)
        
        # Save results
        self.save_results(figures, analysis, timestamp)
        
        # Show plots
        print(f"\n🎉 Visualization completed!")
        print(f"📊 Generated {len(figures)} visualizations")
        print(f"🔍 Embedding quality analysis completed")
        
        if HAS_PLOTLY:
            print(f"💡 Interactive Plotly visualizations could be added for better exploration")
        
        plt.show()
        
        return figures, analysis


def main():
    """Run the embedding visualization."""
    
    try:
        visualizer = EmbeddingVisualizer()
        
        # Configure visualization
        methods = []
        if HAS_UMAP:
            methods.append('umap')
        if HAS_SKLEARN:
            methods.append('tsne')
        methods.append('pca')  # Always available
        
        # Run visualization
        figures, analysis = visualizer.run_visualization(
            max_items=800,
            methods=methods[:2]  # Use top 2 methods to avoid too many plots
        )
        
        print(f"\n✅ Embedding visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()