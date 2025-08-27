import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
import os

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower
from src.inference.faiss_index import FAISSItemIndex
from src.preprocessing.data_loader import DataProcessor


class RecommendationEngine:
    """Complete recommendation engine using trained two-tower model."""
    
    def __init__(self, artifacts_path: str = "src/artifacts/"):
        self.artifacts_path = artifacts_path
        self.item_tower = None
        self.user_tower = None
        self.rating_model = None
        self.faiss_index = None
        self.data_processor = None
        self.items_df = None
        self.income_thresholds = None  # Store income thresholds for categorization
        
        # Load components
        self._load_all_components()
    
    def categorize_age(self, age: float) -> int:
        """Categorize age into 6 demographic groups."""
        if age < 18:
            return 0  # Teen
        elif age < 26:
            return 1  # Young Adult
        elif age < 36:
            return 2  # Adult
        elif age < 51:
            return 3  # Middle Age
        elif age < 66:
            return 4  # Mature
        else:
            return 5  # Senior
    
    def categorize_income(self, income: float) -> int:
        """Categorize income based on training set percentiles."""
        if self.income_thresholds is None:
            # Default categorization if thresholds not available
            if income < 30000:
                return 0
            elif income < 50000:
                return 1
            elif income < 75000:
                return 2
            elif income < 100000:
                return 3
            else:
                return 4
        
        # Use stored percentile thresholds from training
        category = np.digitize([income], self.income_thresholds[1:-1])[0]
        return min(max(category, 0), 4)
    
    def _load_all_components(self):
        """Load all required components for inference."""
        
        print("Loading recommendation engine components...")
        
        # Load data processor and vocabularies
        self.data_processor = DataProcessor()
        self.data_processor.load_vocabularies(f"{self.artifacts_path}/vocabularies.pkl")
        
        # Load items dataframe for metadata
        self.items_df = pd.read_csv("datasets/items.csv")
        
        # Load trained models
        self._load_item_tower()
        self._load_user_tower()
        self._load_rating_model()
        
        # Load FAISS index
        self.faiss_index = FAISSItemIndex()
        self.faiss_index.load_index(self.artifacts_path)
        
        print("All components loaded successfully!")
    
    def _load_item_tower(self):
        """Load trained item tower."""
        
        # Read config
        with open(f"{self.artifacts_path}/item_tower_config.txt", 'r') as f:
            config = {}
            for line in f:
                key, value = line.strip().split(': ')
                if key in ['embedding_dim', 'dropout_rate']:
                    config[key] = float(value) if '.' in value else int(value)
                elif key == 'hidden_dims':
                    config[key] = eval(value)
        
        # Build item tower
        self.item_tower = ItemTower(
            item_vocab_size=len(self.data_processor.item_vocab),
            category_vocab_size=len(self.data_processor.category_vocab),
            brand_vocab_size=len(self.data_processor.brand_vocab),
            **config
        )
        
        # Load weights (try fine-tuned first, then pre-trained)
        dummy_input = {
            'product_id': tf.constant([0]),
            'category_id': tf.constant([0]),
            'brand_id': tf.constant([0]),
            'price': tf.constant([0.0])
        }
        _ = self.item_tower(dummy_input)
        
        try:
            self.item_tower.load_weights(f"{self.artifacts_path}/item_tower_weights_finetuned_best")
            print("Loaded fine-tuned item tower")
        except:
            try:
                self.item_tower.load_weights(f"{self.artifacts_path}/item_tower_weights")
                print("Loaded pre-trained item tower")
            except:
                print("Warning: Could not load item tower weights")
    
    def _load_user_tower(self):
        """Load trained user tower."""
        
        self.user_tower = UserTower(
            max_history_length=50,
            embedding_dim=128,  # Changed from 64 to 128
            hidden_dims=[128, 64],  # Match training architecture
            dropout_rate=0.2
        )
        
        # Build user tower with dummy categorical input
        dummy_input = {
            'age': tf.constant([2]),  # Adult category (26-35)
            'gender': tf.constant([1]),  # Male
            'income': tf.constant([2]),  # Middle income category
            'item_history_embeddings': tf.constant([[[0.0] * 128] * 50])  # Changed from 64 to 128
        }
        _ = self.user_tower(dummy_input)
        
        # Adapt normalization layers with training data
        try:
            # Load training data to get income thresholds for categorization
            import pickle
            with open(f"{self.artifacts_path}/training_features.pkl", 'rb') as f:
                training_features = pickle.load(f)
            
            # Note: Training features now contain categorical age/income
            # If we need raw values for threshold calculation, load from original users data
            try:
                users_df = pd.read_csv("datasets/users.csv")
                percentiles = [0, 20, 40, 60, 80, 100]
                self.income_thresholds = np.percentile(users_df['income'], percentiles)
                print(f"Loaded income thresholds: {self.income_thresholds}")
            except Exception as e:
                print(f"Warning: Could not load income thresholds: {e}")
            
            print("Using categorical age and income features")
        except Exception as e:
            print(f"Warning: Could not load training features: {e}")
        
        try:
            self.user_tower.load_weights(f"{self.artifacts_path}/user_tower_weights_best")
            print("Loaded trained user tower")
        except:
            print("Warning: Could not load user tower weights")
    
    def _load_rating_model(self):
        """Load trained rating prediction model."""
        
        # Create rating model with same architecture as training
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Build model with dummy input (concatenated user and item embeddings)
        dummy_input = tf.constant([[0.0] * 256])  # 128 + 128 = 256
        _ = self.rating_model(dummy_input)
        
        try:
            self.rating_model.load_weights(f"{self.artifacts_path}/rating_model_weights_best")
            print("Loaded trained rating model")
        except:
            print("Warning: Could not load rating model weights")
    
    def prepare_user_features(self,
                            age: int,
                            gender: str,
                            income: float,
                            interaction_history: List[int] = None) -> Dict[str, tf.Tensor]:
        """Prepare user features for inference."""
        
        if interaction_history is None:
            interaction_history = []
        
        # Convert gender
        gender_numeric = 1 if gender.lower() == 'male' else 0
        
        # Categorize age and income
        age_category = self.categorize_age(age)
        income_category = self.categorize_income(income)
        
        # Get item embeddings for history
        history_embeddings = []
        for item_id in interaction_history:
            if item_id in self.faiss_index.item_id_to_idx:
                embedding = self.faiss_index.get_item_embedding(item_id)
                history_embeddings.append(embedding)
            else:
                # Use zero embedding for unknown items
                history_embeddings.append(np.zeros(128))  # Changed from 64 to 128
        
        # Pad or truncate to max_history_length
        max_history_length = 50
        if len(history_embeddings) < max_history_length:
            # Add padding at the END so real interactions are at the BEGINNING
            padding = [np.zeros(128)] * (max_history_length - len(history_embeddings))
            history_embeddings = history_embeddings + padding
        else:
            # Keep most recent interactions
            history_embeddings = history_embeddings[-max_history_length:]
        
        history_embeddings = np.array(history_embeddings, dtype=np.float32)
        
        # Prepare features with categorical demographics
        user_features = {
            'age': tf.constant([age_category]),  # Categorical age (0-5)
            'gender': tf.constant([gender_numeric]),  # Categorical gender (0-1)
            'income': tf.constant([income_category]),  # Categorical income (0-4)
            'item_history_embeddings': tf.constant([history_embeddings])
        }
        
        return user_features
    
    def prepare_item_features(self, item_ids: List[int]) -> Dict[str, tf.Tensor]:
        """Prepare item features for inference."""
        
        features = {
            'product_id': [],
            'category_id': [],
            'brand_id': [],
            'price': []
        }
        
        for item_id in item_ids:
            # Find item in dataframe
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            
            if len(item_row) > 0:
                item_row = item_row.iloc[0]
                
                features['product_id'].append(self.data_processor.item_vocab.get(item_id, 0))
                features['category_id'].append(self.data_processor.category_vocab.get(item_row['category_id'], 0))
                features['brand_id'].append(self.data_processor.brand_vocab.get(item_row['brand'], 0))
                features['price'].append(float(item_row['price']))
            else:
                # Unknown item
                features['product_id'].append(0)
                features['category_id'].append(0)
                features['brand_id'].append(0)
                features['price'].append(0.0)
        
        # Convert to tensors
        return {k: tf.constant(v) for k, v in features.items()}
    
    def get_user_embedding(self, 
                          age: int,
                          gender: str,
                          income: float,
                          interaction_history: List[int] = None) -> np.ndarray:
        """Get user embedding from user tower."""
        
        user_features = self.prepare_user_features(age, gender, income, interaction_history)
        user_embedding = self.user_tower(user_features, training=False)
        
        return user_embedding.numpy()[0]
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get item embedding from FAISS index or item tower."""
        
        # First try FAISS index (faster)
        embedding = self.faiss_index.get_item_embedding(item_id)
        if embedding is not None:
            return embedding
        
        # Fall back to item tower for new items
        item_features = self.prepare_item_features([item_id])
        item_embedding = self.item_tower(item_features, training=False)
        
        return item_embedding.numpy()[0]
    
    def recommend_items_collaborative(self,
                                   age: int,
                                   gender: str,
                                   income: float,
                                   interaction_history: List[int] = None,
                                   k: int = 10,
                                   exclude_history: bool = True,
                                   category_boost: float = 1.3) -> List[Tuple[int, float, Dict]]:
        """Generate recommendations using collaborative filtering with category awareness."""
        
        # Get user embedding
        user_embedding = self.get_user_embedding(age, gender, income, interaction_history)
        
        # Find similar items using FAISS (get more candidates for boosting)
        similar_items = self.faiss_index.search_by_embedding(user_embedding, k * 4)
        
        # Get user's preferred categories from interaction history
        user_categories = set()
        if interaction_history:
            for item_id in interaction_history[-10:]:  # Focus on recent interactions
                item_row = self.items_df[self.items_df['product_id'] == item_id]
                if len(item_row) > 0:
                    user_categories.add(item_row.iloc[0]['category_code'])
        
        # Filter out interaction history and apply category boosting
        boosted_items = []
        history_set = set(interaction_history) if (exclude_history and interaction_history) else set()
        
        for item_id, score in similar_items:
            if item_id in history_set:
                continue
                
            # Get item category
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                item_category = item_row.iloc[0]['category_code']
                
                # Boost score if item is in user's preferred categories
                if item_category in user_categories:
                    boosted_score = score * category_boost
                else:
                    boosted_score = score
                
                boosted_items.append((item_id, boosted_score))
        
        # Sort by boosted score and take top k
        boosted_items.sort(key=lambda x: x[1], reverse=True)
        boosted_items = boosted_items[:k]
        
        # Add item metadata
        recommendations = []
        for item_id, score in boosted_items:
            item_info = self._get_item_info(item_id)
            recommendations.append((item_id, score, item_info))
        
        return recommendations
    
    def recommend_items_content_based(self,
                                    seed_item_id: int,
                                    k: int = 10,
                                    same_category_ratio: float = None) -> List[Tuple[int, float, Dict]]:
        """Generate recommendations using content-based filtering with optional category constraint."""
        
        if same_category_ratio is None:
            # Original behavior - pure similarity ranking
            similar_items = self.faiss_index.search_similar_items(seed_item_id, k)
            recommendations = []
            for item_id, score in similar_items:
                item_info = self._get_item_info(item_id)
                recommendations.append((item_id, score, item_info))
            return recommendations
        
        else:
            # Category-aware similar items for clicked recommendations
            print(f"Finding similar items with {same_category_ratio*100}% same-category constraint")
            
            # Get seed item category
            seed_item_info = self._get_item_info(seed_item_id)
            seed_category = seed_item_info.get('category_code', '')
            print(f"Seed item {seed_item_id} category: {seed_category}")
            
            # Get more candidates (3x) to ensure category diversity
            candidate_items = self.faiss_index.search_similar_items(seed_item_id, k * 3)
            print(f"Retrieved {len(candidate_items)} candidates from FAISS")
            
            # Separate by category
            same_category_items = []
            different_category_items = []
            
            for item_id, score in candidate_items:
                item_info = self._get_item_info(item_id)
                item_category = item_info.get('category_code', '')
                
                if item_category == seed_category:
                    same_category_items.append((item_id, score, item_info))
                else:
                    different_category_items.append((item_id, score, item_info))
            
            print(f"Same category items: {len(same_category_items)}, Different category: {len(different_category_items)}")
            
            # Calculate target counts (60/40 split)
            same_category_count = int(k * same_category_ratio)  # 6 out of 10
            different_category_count = k - same_category_count   # 4 out of 10
            
            print(f"Target: {same_category_count} same category, {different_category_count} different category")
            
            # Build balanced recommendation list
            recommendations = []
            
            # Add same-category items (up to 60%)
            recommendations.extend(same_category_items[:same_category_count])
            
            # Add different-category items (up to 40%)
            recommendations.extend(different_category_items[:different_category_count])
            
            # Fill any remaining slots with best available items
            if len(recommendations) < k:
                remaining_items = same_category_items[same_category_count:] + different_category_items[different_category_count:]
                remaining_items.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
                needed = k - len(recommendations)
                recommendations.extend(remaining_items[:needed])
            
            print(f"Final recommendations: {len(recommendations)} items")
            
            # Log category distribution for verification
            final_same_category = sum(1 for _, _, item_info in recommendations if item_info.get('category_code', '') == seed_category)
            print(f"Final category distribution: {final_same_category}/{len(recommendations)} same category ({final_same_category/len(recommendations)*100:.1f}%)")
            
            return recommendations[:k]
    
    def recommend_items_hybrid(self,
                             age: int,
                             gender: str,
                             income: float,
                             interaction_history: List[int] = None,
                             k: int = 10,
                             collaborative_weight: float = 0.7) -> List[Tuple[int, float, Dict]]:
        """Generate hybrid recommendations combining collaborative and content-based."""
        
        # Get collaborative recommendations
        collab_recs = self.recommend_items_collaborative(
            age, gender, income, interaction_history, k * 2
        )
        
        # Get content-based recommendations from recent interactions
        content_recs = []
        if interaction_history:
            # Use most recent item as seed
            recent_item = interaction_history[-1]
            content_recs = self.recommend_items_content_based(recent_item, k)
        
        # Combine recommendations with weighted scores
        item_scores = {}
        
        # Add collaborative scores
        for item_id, score, info in collab_recs:
            item_scores[item_id] = {
                'collab_score': score,
                'content_score': 0.0,
                'info': info
            }
        
        # Add content-based scores
        for item_id, score, info in content_recs:
            if item_id in item_scores:
                item_scores[item_id]['content_score'] = score
            else:
                item_scores[item_id] = {
                    'collab_score': 0.0,
                    'content_score': score,
                    'info': info
                }
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        for item_id, scores in item_scores.items():
            hybrid_score = (
                collaborative_weight * scores['collab_score'] +
                (1 - collaborative_weight) * scores['content_score']
            )
            hybrid_recommendations.append((item_id, hybrid_score, scores['info']))
        
        # Sort by hybrid score and take top k
        hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_recommendations[:k]
    
    def _get_item_info(self, item_id: int) -> Dict:
        """Get item metadata."""
        
        item_row = self.items_df[self.items_df['product_id'] == item_id]
        
        if len(item_row) > 0:
            item_row = item_row.iloc[0]
            return {
                'product_id': int(item_id),
                'category_id': int(item_row['category_id']),
                'category_code': str(item_row['category_code']),
                'brand': str(item_row['brand']) if pd.notna(item_row['brand']) else 'Unknown',
                'price': float(item_row['price'])
            }
        else:
            return {
                'product_id': int(item_id),
                'category_id': 0,
                'category_code': 'unknown',
                'brand': 'Unknown',
                'price': 0.0
            }
    
    def predict_rating(self,
                      age: int,
                      gender: str,
                      income: float,
                      item_id: int,
                      interaction_history: List[int] = None) -> float:
        """Predict rating for a specific user-item pair."""
        
        if self.rating_model is None:
            return 0.5  # Default prediction
        
        # Prepare user features
        user_features = self.prepare_user_features(age, gender, income, interaction_history)
        
        # Prepare item features
        if item_id not in self.data_processor.item_vocab:
            return 0.5  # Unknown item
        
        item_features = self.prepare_item_features([item_id])
        
        # Get embeddings
        user_embedding = self.user_tower(user_features, training=False)
        item_embedding = self.item_tower(item_features, training=False)
        
        # Concatenate embeddings
        concatenated = tf.concat([user_embedding, item_embedding], axis=-1)
        
        # Predict rating
        rating_prediction = self.rating_model(concatenated, training=False)
        
        return float(rating_prediction.numpy()[0][0])


def main():
    """Demo the recommendation engine."""
    
    # Initialize recommendation engine
    print("Initializing recommendation engine...")
    engine = RecommendationEngine()
    
    # Demo user profile
    demo_user = {
        'age': 32,
        'gender': 'male',
        'income': 75000,
        'interaction_history': [1000978, 1001588, 1001618]  # Sample item IDs
    }
    
    print(f"\nDemo user profile:")
    print(f"Age: {demo_user['age']}")
    print(f"Gender: {demo_user['gender']}")
    print(f"Income: ${demo_user['income']:,}")
    print(f"Interaction history: {demo_user['interaction_history']}")
    
    # Generate collaborative recommendations
    print("\n=== Collaborative Filtering Recommendations ===")
    collab_recs = engine.recommend_items_collaborative(**demo_user, k=5)
    
    for i, (item_id, score, info) in enumerate(collab_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    # Generate content-based recommendations
    print("\n=== Content-Based Recommendations (similar to recent item) ===")
    if demo_user['interaction_history']:
        content_recs = engine.recommend_items_content_based(
            seed_item_id=demo_user['interaction_history'][-1], k=5
        )
        
        for i, (item_id, score, info) in enumerate(content_recs, 1):
            print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    # Generate hybrid recommendations
    print("\n=== Hybrid Recommendations ===")
    hybrid_recs = engine.recommend_items_hybrid(**demo_user, k=5)
    
    for i, (item_id, score, info) in enumerate(hybrid_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    print("\nRecommendation engine demo completed!")


if __name__ == "__main__":
    main()