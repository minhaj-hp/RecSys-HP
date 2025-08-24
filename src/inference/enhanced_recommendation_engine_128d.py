#!/usr/bin/env python3
"""
Enhanced recommendation engine using 128D embeddings with diversity regularization.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.enhanced_two_tower import EnhancedItemTower, EnhancedUserTower
from src.inference.faiss_index import FAISSItemIndex
from src.preprocessing.data_loader import DataProcessor
from src.preprocessing.user_data_preparation import prepare_user_features
from src.utils.real_user_selector import RealUserSelector


class Enhanced128DRecommendationEngine:
    """Enhanced recommendation engine with 128D embeddings and all improvements."""
    
    def __init__(self, artifacts_path: str = "src/artifacts/"):
        self.artifacts_path = artifacts_path
        self.embedding_dim = 128  # Fixed to 128D
        
        # Model components
        self.item_tower = None
        self.user_tower = None
        self.rating_model = None
        self.faiss_index = None
        self.data_processor = None
        
        # Data
        self.items_df = None
        self.users_df = None
        self.income_thresholds = None
        
        # Load all components
        self._load_all_components()
    
    def _load_all_components(self):
        """Load all enhanced model components."""
        
        print("Loading enhanced 128D recommendation engine...")
        
        # Load data processor
        self.data_processor = DataProcessor()
        try:
            self.data_processor.load_vocabularies(f"{self.artifacts_path}/vocabularies.pkl")
        except FileNotFoundError:
            print("❌ Vocabularies not found. Please train the model first.")
            return
        
        # Load datasets
        self.items_df = pd.read_csv("datasets/items.csv")
        self.users_df = pd.read_csv("datasets/users.csv")
        
        # Load enhanced model components
        self._load_enhanced_models()
        
        # Load FAISS index with 128D
        try:
            self.faiss_index = FAISSItemIndex(embedding_dim=self.embedding_dim)
            # Try to load enhanced embeddings first
            if os.path.exists(f"{self.artifacts_path}/enhanced_item_embeddings.npy"):
                enhanced_embeddings = np.load(
                    f"{self.artifacts_path}/enhanced_item_embeddings.npy", 
                    allow_pickle=True
                ).item()
                self.faiss_index.build_index(enhanced_embeddings)
                print("✅ Loaded enhanced 128D FAISS index")
            else:
                print("⚠️  Enhanced embeddings not found. Train enhanced model first.")
                self.faiss_index = None
        except Exception as e:
            print(f"⚠️  Could not load FAISS index: {e}")
            self.faiss_index = None
        
        # Load income thresholds for categorical demographics
        self._load_income_thresholds()
        
        print("✅ Enhanced 128D engine loaded successfully!")
    
    def _load_enhanced_models(self):
        """Load enhanced model components."""
        
        try:
            # Create model architecture
            self.item_tower = EnhancedItemTower(
                item_vocab_size=len(self.data_processor.item_vocab),
                category_vocab_size=len(self.data_processor.category_vocab),
                brand_vocab_size=len(self.data_processor.brand_vocab),
                embedding_dim=self.embedding_dim,
                use_bias=True,
                use_diversity_reg=False  # Disable during inference
            )
            
            self.user_tower = EnhancedUserTower(
                max_history_length=50,
                embedding_dim=self.embedding_dim,
                use_bias=True,
                use_diversity_reg=False  # Disable during inference
            )
            
            # Create rating model
            self.rating_model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            
            # Load weights - try enhanced first, fall back to regular
            model_files = [
                ('enhanced_item_tower_weights_enhanced_best', 'enhanced_user_tower_weights_enhanced_best', 'enhanced_rating_model_weights_enhanced_best'),
                ('enhanced_item_tower_weights_enhanced_final', 'enhanced_user_tower_weights_enhanced_final', 'enhanced_rating_model_weights_enhanced_final'),
            ]
            
            loaded = False
            for item_file, user_file, rating_file in model_files:
                try:
                    # Need to build models first with dummy data
                    self._build_models()
                    
                    self.item_tower.load_weights(f"{self.artifacts_path}/{item_file}")
                    self.user_tower.load_weights(f"{self.artifacts_path}/{user_file}")
                    self.rating_model.load_weights(f"{self.artifacts_path}/{rating_file}")
                    
                    print(f"✅ Loaded enhanced model: {item_file}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Could not load {item_file}: {e}")
                    continue
            
            if not loaded:
                print("❌ No enhanced model weights found. Please train enhanced model first.")
                self.item_tower = None
                self.user_tower = None
                self.rating_model = None
                
        except Exception as e:
            print(f"❌ Failed to load enhanced models: {e}")
            self.item_tower = None
            self.user_tower = None
            self.rating_model = None
    
    def _build_models(self):
        """Build models with dummy data to initialize weights."""
        
        # Dummy item features
        dummy_item_features = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([100.0])
        }
        
        # Dummy user features  
        dummy_user_features = {
            'age': tf.constant([2]),  # Adult category
            'gender': tf.constant([0]),  # Female
            'income': tf.constant([2]),  # Middle income
            'item_history_embeddings': tf.constant(np.zeros((1, 50, self.embedding_dim), dtype=np.float32))
        }
        
        # Forward pass to build models
        _ = self.item_tower(dummy_item_features, training=False)
        _ = self.user_tower(dummy_user_features, training=False)
        
        # Build rating model
        dummy_concat = tf.constant(np.zeros((1, self.embedding_dim * 2), dtype=np.float32))
        _ = self.rating_model(dummy_concat, training=False)
    
    def _load_income_thresholds(self):
        """Load income thresholds for categorical processing."""
        
        # Calculate income thresholds from training data
        user_incomes = self.users_df['income'].values
        self.income_thresholds = np.percentile(user_incomes, [0, 20, 40, 60, 80, 100])
        print(f"Income thresholds: {self.income_thresholds}")
    
    def categorize_age(self, age: float) -> int:
        """Categorize age into 6 groups."""
        if age < 18: return 0      # Teen
        elif age < 26: return 1    # Young Adult  
        elif age < 36: return 2    # Adult
        elif age < 51: return 3    # Middle Age
        elif age < 66: return 4    # Mature
        else: return 5             # Senior
    
    def categorize_income(self, income: float) -> int:
        """Categorize income into 5 percentile groups."""
        category = np.digitize([income], self.income_thresholds[1:-1])[0]
        return min(max(category, 0), 4)
    
    def categorize_gender(self, gender: str) -> int:
        """Categorize gender."""
        return 1 if gender.lower() == 'male' else 0
    
    def get_user_embedding(self, 
                          age: int, 
                          gender: str, 
                          income: float,
                          interaction_history: List[int] = None) -> np.ndarray:
        """Generate user embedding with categorical demographics."""
        
        if self.user_tower is None:
            print("❌ User tower not loaded")
            return None
        
        # Categorize demographics
        age_cat = self.categorize_age(age)
        gender_cat = self.categorize_gender(gender)
        income_cat = self.categorize_income(income)
        
        # Prepare interaction history embeddings
        if interaction_history is None:
            interaction_history = []
        
        # Get item embeddings for history
        history_embeddings = np.zeros((50, self.embedding_dim), dtype=np.float32)
        
        for i, item_id in enumerate(interaction_history[:50]):
            if self.faiss_index and item_id in self.faiss_index.item_id_to_idx:
                item_emb = self.faiss_index.get_item_embedding(item_id)
                if item_emb is not None:
                    history_embeddings[i] = item_emb
        
        # Create user features
        user_features = {
            'age': tf.constant([age_cat]),
            'gender': tf.constant([gender_cat]),
            'income': tf.constant([income_cat]),
            'item_history_embeddings': tf.constant([history_embeddings])
        }
        
        # Get embedding
        user_output = self.user_tower(user_features, training=False)
        if isinstance(user_output, tuple):
            user_embedding = user_output[0].numpy()[0]
        else:
            user_embedding = user_output.numpy()[0]
        
        return user_embedding
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get item embedding."""
        
        if self.faiss_index:
            return self.faiss_index.get_item_embedding(item_id)
        
        # Fallback to model computation
        if self.item_tower is None:
            return None
        
        item_row = self.items_df[self.items_df['product_id'] == item_id]
        if item_row.empty:
            return None
        
        item_data = item_row.iloc[0]
        
        # Prepare features
        item_features = {
            'product_id': tf.constant([self.data_processor.item_vocab.get(item_id, 0)]),
            'category_id': tf.constant([self.data_processor.category_vocab.get(item_data['category_id'], 0)]),
            'brand_id': tf.constant([self.data_processor.brand_vocab.get(item_data.get('brand', 'unknown'), 0)]),
            'price': tf.constant([float(item_data.get('price', 0.0))])
        }
        
        # Get embedding
        item_output = self.item_tower(item_features, training=False)
        if isinstance(item_output, tuple):
            item_embedding = item_output[0].numpy()[0]
        else:
            item_embedding = item_output.numpy()[0]
        
        return item_embedding
    
    def recommend_items_enhanced(self,
                               age: int,
                               gender: str, 
                               income: float,
                               interaction_history: List[int] = None,
                               k: int = 10,
                               diversity_weight: float = 0.3,
                               category_boost: float = 1.5) -> List[Tuple[int, float, Dict]]:
        """Generate enhanced recommendations with diversity and category boosting."""
        
        if not self.faiss_index:
            print("❌ FAISS index not available")
            return []
        
        # Get user embedding
        user_embedding = self.get_user_embedding(age, gender, income, interaction_history)
        if user_embedding is None:
            return []
        
        # Get candidate recommendations (more than needed for filtering)
        candidates = self.faiss_index.search_by_embedding(user_embedding, k * 3)
        
        # Filter out items from interaction history
        if interaction_history:
            history_set = set(interaction_history)
            candidates = [(item_id, score) for item_id, score in candidates 
                         if item_id not in history_set]
        
        # Add item metadata and apply enhancements
        enhanced_candidates = []
        
        for item_id, similarity_score in candidates[:k * 2]:
            # Get item info
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if item_row.empty:
                continue
            
            item_info = item_row.iloc[0].to_dict()
            
            # Enhanced scoring with multiple factors
            final_score = similarity_score
            
            # Category boosting based on user history
            if interaction_history and category_boost > 1.0:
                user_categories = self._get_user_categories(interaction_history)
                item_category = item_info.get('category_code', '')
                
                if item_category in user_categories:
                    category_preference = user_categories[item_category]
                    final_score *= (1 + (category_boost - 1) * category_preference)
            
            enhanced_candidates.append((item_id, final_score, item_info))
        
        # Sort by enhanced scores
        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering
        if diversity_weight > 0:
            diversified_candidates = self._apply_diversity_filter(
                enhanced_candidates, diversity_weight
            )
        else:
            diversified_candidates = enhanced_candidates
        
        return diversified_candidates[:k]
    
    def _get_user_categories(self, interaction_history: List[int]) -> Dict[str, float]:
        """Get user's category preferences from history."""
        
        category_counts = Counter()
        
        for item_id in interaction_history:
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if not item_row.empty:
                category = item_row.iloc[0].get('category_code', 'Unknown')
                category_counts[category] += 1
        
        # Convert to preferences (percentages)
        total = sum(category_counts.values())
        if total == 0:
            return {}
        
        return {cat: count / total for cat, count in category_counts.items()}
    
    def _apply_diversity_filter(self, 
                              candidates: List[Tuple[int, float, Dict]], 
                              diversity_weight: float,
                              max_per_category: int = 3) -> List[Tuple[int, float, Dict]]:
        """Apply diversity filtering to recommendations."""
        
        category_counts = defaultdict(int)
        diversified = []
        
        for item_id, score, item_info in candidates:
            category = item_info.get('category_code', 'Unknown')
            
            # Apply diversity penalty
            if category_counts[category] >= max_per_category:
                # Penalty for over-representation
                diversity_penalty = diversity_weight * (category_counts[category] - max_per_category + 1)
                adjusted_score = score * (1 - diversity_penalty)
            else:
                adjusted_score = score
            
            diversified.append((item_id, adjusted_score, item_info))
            category_counts[category] += 1
        
        # Re-sort by adjusted scores
        diversified.sort(key=lambda x: x[1], reverse=True)
        return diversified
    
    def predict_rating(self,
                      age: int,
                      gender: str,
                      income: float, 
                      item_id: int,
                      interaction_history: List[int] = None) -> float:
        """Predict rating for user-item pair."""
        
        if self.rating_model is None:
            return 0.5  # Default rating
        
        # Get embeddings
        user_embedding = self.get_user_embedding(age, gender, income, interaction_history)
        item_embedding = self.get_item_embedding(item_id)
        
        if user_embedding is None or item_embedding is None:
            return 0.5
        
        # Concatenate embeddings
        combined = np.concatenate([user_embedding, item_embedding])
        combined = tf.constant([combined])
        
        # Predict rating
        rating = self.rating_model(combined, training=False)
        return float(rating.numpy()[0][0])


def demo_enhanced_engine():
    """Demo the enhanced 128D recommendation engine."""
    
    print("🚀 ENHANCED 128D RECOMMENDATION ENGINE DEMO")
    print("="*70)
    
    try:
        # Initialize engine
        engine = Enhanced128DRecommendationEngine()
        
        if engine.item_tower is None:
            print("❌ Enhanced model not available. Please train first using:")
            print("   python train_enhanced_model.py")
            return
        
        # Get real user for testing
        real_user_selector = RealUserSelector()
        test_users = real_user_selector.get_real_users(n=2, min_interactions=10)
        
        for user in test_users:
            print(f"\n📊 Testing User {user['user_id']} ({user['age']}yr {user['gender']}):")
            print(f"   Income: ${user['income']:,}")
            print(f"   History: {len(user['interaction_history'])} items")
            
            # Test enhanced recommendations
            try:
                recs = engine.recommend_items_enhanced(
                    age=user['age'],
                    gender=user['gender'],
                    income=user['income'],
                    interaction_history=user['interaction_history'][:20],
                    k=10,
                    diversity_weight=0.3,
                    category_boost=1.5
                )
                
                print(f"   🎯 Enhanced Recommendations:")
                categories = []
                for i, (item_id, score, item_info) in enumerate(recs[:5]):
                    category = item_info.get('category_code', 'Unknown')[:30]
                    price = item_info.get('price', 0)
                    categories.append(category)
                    print(f"      #{i+1} Item {item_id}: {score:.4f} | ${price:.2f} | {category}")
                
                # Analyze diversity
                unique_categories = len(set(categories))
                print(f"   📈 Diversity: {unique_categories}/{len(categories)} unique categories")
                
                # Test rating prediction
                if recs:
                    test_item = recs[0][0]
                    predicted_rating = engine.predict_rating(
                        age=user['age'],
                        gender=user['gender'],
                        income=user['income'],
                        item_id=test_item,
                        interaction_history=user['interaction_history'][:20]
                    )
                    print(f"   ⭐ Rating prediction for item {test_item}: {predicted_rating:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Enhanced 128D engine demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_enhanced_engine()