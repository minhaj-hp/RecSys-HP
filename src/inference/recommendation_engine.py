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
    
    def categorize_profession(self, profession: str) -> int:
        """Categorize profession into numeric categories."""
        profession_map = {
            "Technology": 0,
            "Healthcare": 1, 
            "Education": 2,
            "Finance": 3,
            "Retail": 4,
            "Manufacturing": 5,
            "Services": 6,
            "Other": 7
        }
        return profession_map.get(profession, 7)  # Default to "Other"
    
    def categorize_location(self, location: str) -> int:
        """Categorize location into numeric categories."""
        location_map = {
            "Urban": 0,
            "Suburban": 1,
            "Rural": 2
        }
        return location_map.get(location, 0)  # Default to "Urban"
    
    def categorize_education_level(self, education: str) -> int:
        """Categorize education level into numeric categories."""
        education_map = {
            "High School": 0,
            "Some College": 1,
            "Bachelor's": 2,
            "Master's": 3,
            "PhD+": 4
        }
        return education_map.get(education, 0)  # Default to "High School"
    
    def categorize_marital_status(self, marital_status: str) -> int:
        """Categorize marital status into numeric categories."""
        marital_map = {
            "Single": 0,
            "Married": 1,
            "Divorced": 2,
            "Widowed": 3
        }
        return marital_map.get(marital_status, 0)  # Default to "Single"
    
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
            category_code_vocab_size=len(self.data_processor.category_vocab),  # Use same size as category vocab
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
            'profession': tf.constant([0]),  # Technology
            'location': tf.constant([0]),  # Urban
            'education_level': tf.constant([2]),  # Bachelor's
            'marital_status': tf.constant([1]),  # Married
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
            print("✅ Loaded trained user tower weights (best model)")
        except Exception as e:
            print(f"❌ Warning: Could not load user tower weights: {e}")
            print("   This is expected if you haven't retrained with the new architecture yet")
            print("   Please run joint training to generate compatible weights")
    
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
                            profession: str = "Other",
                            location: str = "Urban",
                            education_level: str = "High School",
                            marital_status: str = "Single",
                            interaction_history: List[int] = None) -> Dict[str, tf.Tensor]:
        """Prepare user features for inference."""
        
        if interaction_history is None:
            interaction_history = []
        
        # Convert gender
        gender_numeric = 1 if gender.lower() == 'male' else 0
        
        # Categorize all demographics
        age_category = self.categorize_age(age)
        income_category = self.categorize_income(income)
        profession_category = self.categorize_profession(profession)
        location_category = self.categorize_location(location)
        education_category = self.categorize_education_level(education_level)
        marital_category = self.categorize_marital_status(marital_status)
        
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
            'profession': tf.constant([profession_category]),  # Categorical profession (0-7)
            'location': tf.constant([location_category]),  # Categorical location (0-2)
            'education_level': tf.constant([education_category]),  # Categorical education (0-4)
            'marital_status': tf.constant([marital_category]),  # Categorical marital status (0-3)
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
                          profession: str = "Other",
                          location: str = "Urban",
                          education_level: str = "High School",
                          marital_status: str = "Single",
                          interaction_history: List[int] = None) -> np.ndarray:
        """Get user embedding from user tower."""
        
        user_features = self.prepare_user_features(age, gender, income, profession, location, education_level, marital_status, interaction_history)
        user_embedding = self.user_tower(user_features, training=False)
        
        return user_embedding.numpy()[0]
    
    def get_user_embedding_enhanced(self, 
                                  age: int,
                                  gender: str, 
                                  income: float,
                                  profession: str = "Other",
                                  location: str = "Urban", 
                                  education_level: str = "High School",
                                  marital_status: str = "Single",
                                  interaction_history: List[int] = None) -> np.ndarray:
        """Enhanced user embedding that handles zero interactions better."""
        
        # Get base embedding
        base_embedding = self.get_user_embedding(
            age, gender, income, profession, location, education_level, marital_status, interaction_history
        )
        
        # Check if this is a zero-interaction user
        has_interactions = interaction_history and len(interaction_history) > 0
        
        if not has_interactions:
            # For zero interactions, amplify the demographic component
            # This is a heuristic fix until we retrain the model
            
            # Create demographic-enhanced embedding
            demographic_mask = np.ones_like(base_embedding)
            
            # Amplify first 50% of dimensions (likely demographic-influenced)
            mid_point = len(base_embedding) // 2
            demographic_mask[:mid_point] *= 3.0  # Strong amplification
            
            # Reduce influence of latter dimensions (likely history-influenced) 
            demographic_mask[mid_point:] *= 0.2  # Strong reduction
            
            enhanced_embedding = base_embedding * demographic_mask
            
            # Add demographic-specific variation to differentiate profiles
            demographic_hash = (
                age * 1000 + 
                (1 if gender.lower() == 'male' else 0) * 100 +
                int(income / 10000) * 10 +
                self.categorize_profession(profession) * 7 +
                self.categorize_location(location) * 3 +
                self.categorize_education_level(education_level) * 5 +
                self.categorize_marital_status(marital_status) * 2
            )
            
            np.random.seed(demographic_hash % 2**32)  # Reproducible noise
            demographic_noise = np.random.normal(0, 0.02, base_embedding.shape)  # Increased noise
            enhanced_embedding += demographic_noise
            
            # Renormalize
            enhanced_embedding = enhanced_embedding / np.linalg.norm(enhanced_embedding)
            
            print(f"Enhanced embedding for zero interactions: age={age}, gender={gender}, profession={profession}")
            
            return enhanced_embedding.astype(np.float32)
        
        return base_embedding
    
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
    
    def recommend_items_raw_two_tower(self,
                                   age: int,
                                   gender: str,
                                   income: float,
                                   profession: str = "Other",
                                   location: str = "Urban",
                                   education_level: str = "High School",
                                   marital_status: str = "Single",
                                   interaction_history: List[int] = None,
                                   k: int = 10,
                                   exclude_history: bool = True,
                                   category_boost: float = 1.6) -> List[Tuple[int, float, Dict]]:
        """Generate recommendations using raw two-tower retrieval with category awareness.
        
        This method computes user embeddings via the User Tower, then finds items with
        highest similarity scores via FAISS search over Item Tower embeddings.
        """
        
        # Get enhanced user embedding from User Tower
        user_embedding = self.get_user_embedding_enhanced(age, gender, income, profession, location, education_level, marital_status, interaction_history)
        
        # Find items with highest similarity scores using FAISS over Item Tower embeddings
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
    
    def recommend_items_collaborative(self, *args, **kwargs) -> List[Tuple[int, float, Dict]]:
        """DEPRECATED: Use recommend_items_raw_two_tower() instead.
        
        This method name was misleading as it doesn't implement true collaborative filtering.
        It performs raw two-tower retrieval (user-item embedding similarity).
        """
        import warnings
        warnings.warn(
            "recommend_items_collaborative() is deprecated and misleading. "
            "Use recommend_items_raw_two_tower() instead. This method performs "
            "raw two-tower retrieval, not collaborative filtering.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.recommend_items_raw_two_tower(*args, **kwargs)
    
    def _aggregate_user_history_embedding(self, 
                                        interaction_history: List[int],
                                        aggregation_method: str = "weighted_mean") -> Optional[np.ndarray]:
        """Aggregate user's interaction history into a single embedding vector."""
        
        if not interaction_history:
            return None
        
        # Get embeddings for items in history
        item_embeddings = []
        valid_items = []
        
        for item_id in interaction_history:
            embedding = self.faiss_index.get_item_embedding(item_id)
            if embedding is not None:
                item_embeddings.append(embedding)
                valid_items.append(item_id)
        
        if not item_embeddings:
            print(f"No valid embeddings found for interaction history: {interaction_history}")
            return None
        
        item_embeddings = np.array(item_embeddings)
        print(f"Aggregating {len(item_embeddings)} item embeddings using {aggregation_method}")
        
        # Apply aggregation method
        if aggregation_method == "mean":
            # Simple mean pooling
            aggregated = np.mean(item_embeddings, axis=0)
            
        elif aggregation_method == "weighted_mean":
            # Weight recent interactions higher (exponential decay)
            weights = np.exp(np.linspace(-1, 0, len(item_embeddings)))  # More recent = higher weight
            weights = weights / np.sum(weights)  # Normalize weights
            aggregated = np.average(item_embeddings, axis=0, weights=weights)
            print(f"Applied weighted mean with weights: {weights[-3:]} (showing last 3)")
            
        elif aggregation_method == "max":
            # Element-wise maximum pooling
            aggregated = np.max(item_embeddings, axis=0)
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # L2 normalize the aggregated embedding
        aggregated = aggregated / np.linalg.norm(aggregated)
        
        return aggregated.astype('float32')
    
    def recommend_items_content_based_from_history(self,
                                                 interaction_history: List[int],
                                                 k: int = 10,
                                                 aggregation_method: str = "weighted_mean",
                                                 same_category_ratio: float = None) -> List[Tuple[int, float, Dict]]:
        """Generate recommendations using content-based filtering from aggregated user history."""
        
        # Aggregate user's interaction history
        aggregated_embedding = self._aggregate_user_history_embedding(
            interaction_history, aggregation_method
        )
        
        if aggregated_embedding is None:
            print("Could not create aggregated embedding from interaction history")
            return []
        
        if same_category_ratio is None:
            # Direct ANN search with aggregated embedding
            similar_items = self.faiss_index.search_by_embedding(aggregated_embedding, k)
            recommendations = []
            
            # Filter out items already in interaction history
            interaction_set = set(interaction_history)
            
            for item_id, score in similar_items:
                if item_id not in interaction_set:  # Exclude already interacted items
                    item_info = self._get_item_info(item_id)
                    recommendations.append((item_id, score, item_info))
                    
                    if len(recommendations) >= k:
                        break
            
            print(f"Found {len(recommendations)} content-based recommendations from aggregated history")
            return recommendations
        
        else:
            # Category-aware approach with aggregated embedding
            print(f"Finding similar items with {same_category_ratio*100}% category constraint from aggregated history")
            
            # Analyze user's category preferences from interaction history
            user_categories = {}
            total_interactions = len(interaction_history)
            
            for item_id in interaction_history:
                item_info = self._get_item_info(item_id)
                category = item_info.get('category_code', '')
                if category:
                    user_categories[category] = user_categories.get(category, 0) + 1
            
            # Convert to percentages
            for category in user_categories:
                user_categories[category] = user_categories[category] / total_interactions
            
            print(f"User category preferences: {user_categories}")
            
            # Get more candidates for category filtering
            candidate_items = self.faiss_index.search_by_embedding(aggregated_embedding, k * 3)
            interaction_set = set(interaction_history)
            
            # Separate by category alignment with user preferences
            preferred_category_items = []
            other_category_items = []
            
            for item_id, score in candidate_items:
                if item_id in interaction_set:
                    continue  # Skip already interacted items
                    
                item_info = self._get_item_info(item_id)
                item_category = item_info.get('category_code', '')
                
                # Check if item category matches user's preferred categories
                if item_category in user_categories:
                    preferred_category_items.append((item_id, score, item_info))
                else:
                    other_category_items.append((item_id, score, item_info))
            
            # Calculate target distribution
            preferred_count = int(k * same_category_ratio)
            other_count = k - preferred_count
            
            print(f"Target: {preferred_count} from preferred categories, {other_count} for exploration")
            
            # Build balanced recommendations
            recommendations = []
            recommendations.extend(preferred_category_items[:preferred_count])
            recommendations.extend(other_category_items[:other_count])
            
            # Fill remaining slots with best available items
            if len(recommendations) < k:
                remaining_items = (preferred_category_items[preferred_count:] + 
                                 other_category_items[other_count:])
                remaining_items.sort(key=lambda x: x[1], reverse=True)  # Sort by score
                needed = k - len(recommendations)
                recommendations.extend(remaining_items[:needed])
            
            print(f"Final recommendations: {len(recommendations)} items")
            return recommendations[:k]
    
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
                             profession: str = "Other",
                             location: str = "Urban",
                             education_level: str = "High School",
                             marital_status: str = "Single",
                             interaction_history: List[int] = None,
                             k: int = 10,
                             collaborative_weight: float = 0.7) -> List[Tuple[int, float, Dict]]:
        """Generate hybrid recommendations combining collaborative and content-based."""
        
        # Get collaborative recommendations
        collab_recs = self.recommend_items_collaborative(
            age, gender, income, profession, location, education_level, marital_status, interaction_history, k * 2
        )
        
        # Get content-based recommendations from aggregated user history
        content_recs = []
        if interaction_history:
            # Use aggregated history embedding instead of single recent item
            content_recs = self.recommend_items_content_based_from_history(
                interaction_history, k, aggregation_method="weighted_mean"
            )
        
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
    
    def recommend_items_category_boosted(self,
                                       age: int,
                                       gender: str,
                                       income: float,
                                       profession: str = "Other",
                                       location: str = "Urban",
                                       education_level: str = "High School",
                                       marital_status: str = "Single",
                                       interaction_history: List[int] = None,
                                       k: int = 10,
                                       exclude_history: bool = True) -> List[Tuple[int, float, Dict]]:
        """Generate category-boosted recommendations ensuring 50% from user's interacted categories."""
        
        if not interaction_history or len(interaction_history) == 0:
            # Fallback to collaborative filtering if no interaction history
            return self.recommend_items_collaborative(
                age, gender, income, profession, location, education_level, marital_status, 
                interaction_history, k, exclude_history
            )
        
        # Step 1: Calculate category percentages from interaction history
        category_percentages = self._calculate_category_percentages(interaction_history)
        
        if not category_percentages:
            # Fallback if no categories found
            return self.recommend_items_collaborative(
                age, gender, income, profession, location, education_level, marital_status, 
                interaction_history, k, exclude_history
            )
        
        # Step 2: Get enhanced user embedding and do wide search (increased for better subcategory coverage)
        user_embedding = self.get_user_embedding_enhanced(age, gender, income, profession, location, education_level, marital_status, interaction_history)
        similar_items = self.faiss_index.search_by_embedding(user_embedding, k * 10)  # Increased from k*6 to k*10
        
        # Step 3: Organize candidates by subcategory with parent fallback
        category_candidates = {category: [] for category in category_percentages.keys()}
        parent_category_mapping = {}  # Track parent categories for fallback
        other_candidates = []
        history_set = set(interaction_history) if exclude_history else set()
        
        # Build parent category mapping for fallback
        for subcategory in category_percentages.keys():
            if '.' in subcategory:
                parent = subcategory.split('.')[0]
                if parent not in parent_category_mapping:
                    parent_category_mapping[parent] = []
                parent_category_mapping[parent].append(subcategory)
        
        for item_id, score in similar_items:
            if item_id in history_set:
                continue
                
            # Get item category
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                full_item_category = item_row.iloc[0]['category_code']
                
                # Extract 2-level subcategory for matching
                if '.' in full_item_category:
                    category_parts = full_item_category.split('.')
                    if len(category_parts) >= 2:
                        item_subcategory = f"{category_parts[0]}.{category_parts[1]}"
                    else:
                        item_subcategory = category_parts[0]
                else:
                    item_subcategory = full_item_category
                
                # Try exact subcategory match first
                if item_subcategory in category_percentages:
                    category_candidates[item_subcategory].append((item_id, score))
                else:
                    # Fallback: try parent category match
                    parent_category = item_subcategory.split('.')[0] if '.' in item_subcategory else item_subcategory
                    matched = False
                    
                    if parent_category in parent_category_mapping:
                        # Add to the first subcategory of this parent (round-robin could be improved later)
                        target_subcategory = parent_category_mapping[parent_category][0]
                        category_candidates[target_subcategory].append((item_id, score))
                        matched = True
                    
                    if not matched:
                        other_candidates.append((item_id, score))
        
        # Step 4: Calculate target counts for each subcategory (50% distributed proportionally)
        category_target_count = max(1, k // 2)  # At least 50% from user categories
        
        # Calculate proportional distribution with proper rounding
        category_counts = self._calculate_proportional_distribution(
            category_percentages, category_target_count
        )
        
        # Step 5: Select items with round-robin filling and rebalancing
        selected_recommendations = []
        
        # Fill from user's categories with rebalancing for insufficient candidates
        actual_selections = {}
        unused_allocations = {}
        
        for category, target_count in category_counts.items():
            candidates = sorted(category_candidates[category], key=lambda x: x[1], reverse=True)
            available_count = len(candidates)
            selected_count = min(target_count, available_count)
            
            print(f"[DEBUG] Category {category}: target={target_count}, available={available_count}, selected={selected_count}")
            
            actual_selections[category] = selected_count
            if selected_count < target_count:
                unused_allocations[category] = target_count - selected_count
            
            # Select items from this category
            for i in range(selected_count):
                item_id, score = candidates[i]
                item_info = self._get_item_info(item_id)
                selected_recommendations.append((item_id, score, item_info))
        
        # Step 6: Redistribute unused allocations proportionally
        total_unused = sum(unused_allocations.values())
        if total_unused > 0:
            print(f"[DEBUG] Redistributing {total_unused} unused slots")
            
            # Find categories with remaining candidates for redistribution
            categories_with_extras = {}
            for category, candidates in category_candidates.items():
                used_count = actual_selections.get(category, 0)
                available_extras = len(candidates) - used_count
                if available_extras > 0:
                    categories_with_extras[category] = available_extras
            
            # Redistribute based on original proportions and availability
            redistributed = 0
            for category in sorted(categories_with_extras.keys(), key=lambda c: category_percentages.get(c, 0), reverse=True):
                if redistributed >= total_unused:
                    break
                
                extra_slots = min(unused_allocations.get(category, 0) + 1, categories_with_extras[category])
                candidates = sorted(category_candidates[category], key=lambda x: x[1], reverse=True)
                used_count = actual_selections.get(category, 0)
                
                for i in range(used_count, min(used_count + extra_slots, len(candidates))):
                    if redistributed >= total_unused:
                        break
                    item_id, score = candidates[i]
                    item_info = self._get_item_info(item_id)
                    selected_recommendations.append((item_id, score, item_info))
                    redistributed += 1
                    
        # Step 7: Fill remaining slots with diverse recommendations
        remaining_slots = k - len(selected_recommendations)
        if remaining_slots > 0:
            # Collect all unused candidates (both from user categories and other categories)
            all_remaining = []
            
            # Add unused items from user categories
            for category, candidates in category_candidates.items():
                used_count = len([rec for rec in selected_recommendations if rec[2].get('category_code', '').startswith(category.split('.')[0])])
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                for i in range(used_count, len(sorted_candidates)):
                    all_remaining.append(sorted_candidates[i])
            
            # Add items from other categories
            all_remaining.extend(other_candidates)
            
            # Sort by score and take best remaining
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            
            print(f"[DEBUG] Filling {remaining_slots} remaining slots from {len(all_remaining)} candidates")
            
            for i in range(min(remaining_slots, len(all_remaining))):
                item_id, score = all_remaining[i]
                item_info = self._get_item_info(item_id)
                selected_recommendations.append((item_id, score, item_info))
        
        # Step 7: Sort final recommendations by score and return top k
        selected_recommendations.sort(key=lambda x: x[1], reverse=True)
        return selected_recommendations[:k]
    
    def _calculate_category_percentages(self, interaction_history: List[int]) -> Dict[str, float]:
        """Calculate subcategory percentages from interaction history (2-level depth)."""
        if not interaction_history:
            return {}
        
        category_counts = {}
        total_interactions = 0
        
        for item_id in interaction_history:
            item_row = self.items_df[self.items_df['product_id'] == item_id]
            if len(item_row) > 0:
                full_category = item_row.iloc[0]['category_code']
                
                # Use 2-level subcategory (e.g., "computers.components" from "computers.components.memory")
                if '.' in full_category:
                    category_parts = full_category.split('.')
                    if len(category_parts) >= 2:
                        subcategory = f"{category_parts[0]}.{category_parts[1]}"
                    else:
                        subcategory = category_parts[0]  # Fallback to top-level if only one part
                else:
                    subcategory = full_category
                
                category_counts[subcategory] = category_counts.get(subcategory, 0) + 1
                total_interactions += 1
        
        # Convert to percentages
        category_percentages = {}
        for category, count in category_counts.items():
            category_percentages[category] = (count / total_interactions) * 100
        
        return category_percentages
    
    def _calculate_proportional_distribution(self, category_percentages: Dict[str, float], 
                                           total_target: int) -> Dict[str, int]:
        """Calculate proportional distribution with proper rounding and no minimum distortion."""
        if not category_percentages or total_target <= 0:
            return {}
        
        # Calculate raw allocations (without minimum guarantee)
        total_percentage = sum(category_percentages.values())
        raw_allocations = {}
        remainders = {}
        
        for category, percentage in category_percentages.items():
            if total_percentage > 0:
                raw_allocation = (percentage / total_percentage) * total_target
                raw_allocations[category] = int(raw_allocation)  # Floor
                remainders[category] = raw_allocation - int(raw_allocation)  # Remainder
            else:
                raw_allocations[category] = 0
                remainders[category] = 0
        
        # Distribute remaining slots based on largest remainders
        allocated_so_far = sum(raw_allocations.values())
        remaining_slots = total_target - allocated_so_far
        
        # Sort categories by remainder (largest first) to distribute remaining slots
        sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        
        for i in range(remaining_slots):
            if i < len(sorted_by_remainder):
                category_to_increment = sorted_by_remainder[i][0]
                raw_allocations[category_to_increment] += 1
        
        # Filter out zero allocations (no artificial minimum guarantee)
        final_allocations = {cat: count for cat, count in raw_allocations.items() if count > 0}
        
        print(f"[DEBUG] Proportional distribution: target={total_target}, allocations={final_allocations}")
        return final_allocations
    
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
                      profession: str = "Other",
                      location: str = "Urban",
                      education_level: str = "High School",
                      marital_status: str = "Single",
                      interaction_history: List[int] = None) -> float:
        """Predict rating for a specific user-item pair."""
        
        if self.rating_model is None:
            return 0.5  # Default prediction
        
        # Prepare user features
        user_features = self.prepare_user_features(age, gender, income, profession, location, education_level, marital_status, interaction_history)
        
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
        'profession': 'Technology',
        'location': 'Urban',
        'education_level': "Bachelor's",
        'marital_status': 'Married',
        'interaction_history': [1000978, 1001588, 1001618]  # Sample item IDs
    }
    
    print(f"\nDemo user profile:")
    print(f"Age: {demo_user['age']}")
    print(f"Gender: {demo_user['gender']}")
    print(f"Income: ${demo_user['income']:,}")
    print(f"Profession: {demo_user['profession']}")
    print(f"Location: {demo_user['location']}")
    print(f"Education: {demo_user['education_level']}")
    print(f"Marital Status: {demo_user['marital_status']}")
    print(f"Interaction history: {demo_user['interaction_history']}")
    
    # Generate collaborative recommendations
    print("\n=== Collaborative Filtering Recommendations ===") 
    # Extract demographics and history separately to avoid conflicts
    demo_kwargs = {k: v for k, v in demo_user.items() if k != 'interaction_history'}
    collab_recs = engine.recommend_items_collaborative(
        **demo_kwargs, interaction_history=demo_user['interaction_history'], k=5
    )
    
    for i, (item_id, score, info) in enumerate(collab_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    # Generate content-based recommendations from aggregated history
    print("\n=== Content-Based Recommendations (from aggregated user history) ===")
    if demo_user['interaction_history']:
        content_recs = engine.recommend_items_content_based_from_history(
            interaction_history=demo_user['interaction_history'], k=5
        )
        
        for i, (item_id, score, info) in enumerate(content_recs, 1):
            print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    # Generate hybrid recommendations
    print("\n=== Hybrid Recommendations ===")
    hybrid_recs = engine.recommend_items_hybrid(
        **demo_kwargs, interaction_history=demo_user['interaction_history'], k=5
    )
    
    for i, (item_id, score, info) in enumerate(hybrid_recs, 1):
        print(f"{i}. Item {item_id}: {info['brand']} - ${info['price']:.2f} (Score: {score:.4f})")
    
    print("\nRecommendation engine demo completed!")


if __name__ == "__main__":
    main()