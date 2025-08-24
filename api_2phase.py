#!/usr/bin/env python3
"""
API for 2-Phase Trained Recommendation System

This API serves recommendations from a model trained using the 2-phase approach:
1. Pre-trained item tower
2. Joint training with fine-tuned item tower

Usage:
    python api_2phase.py
    
Then access: http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import sys
import pandas as pd

# Add src to path for imports and set working directory
parent_dir = os.path.dirname(__file__)
sys.path.append(parent_dir)
os.chdir(parent_dir)  # Change to project root directory

from src.inference.recommendation_engine import RecommendationEngine
from src.utils.real_user_selector import RealUserSelector

# Initialize FastAPI app
app = FastAPI(
    title="Two-Tower Recommendation API (2-Phase Training)",
    description="API for serving recommendations using a two-tower architecture trained with 2-phase approach",
    version="1.0.0-2phase"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
recommendation_engine = None
enhanced_recommendation_engine = None
real_user_selector = None


# Pydantic models for request/response
class UserProfile(BaseModel):
    age: int
    gender: str  # "male" or "female"
    income: float
    interaction_history: Optional[List[int]] = []


class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    num_recommendations: int = 10
    recommendation_type: str = "hybrid"  # "collaborative", "content", "hybrid", "enhanced", "enhanced_128d", "category_focused"
    collaborative_weight: Optional[float] = 0.7
    category_boost: Optional[float] = 1.5  # For enhanced recommendations
    enable_category_boost: Optional[bool] = True
    enable_diversity: Optional[bool] = True


class ItemSimilarityRequest(BaseModel):
    item_id: int
    num_recommendations: int = 10


class RatingPredictionRequest(BaseModel):
    user_profile: UserProfile
    item_id: int


class ItemInfo(BaseModel):
    product_id: int
    category_id: int
    category_code: str
    brand: str
    price: float


class RecommendationResponse(BaseModel):
    item_id: int
    score: float
    item_info: ItemInfo


class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    user_profile: UserProfile
    recommendation_type: str
    total_count: int
    training_approach: str = "2-phase"


class RatingPredictionResponse(BaseModel):
    user_profile: UserProfile
    item_id: int
    predicted_rating: float
    item_info: ItemInfo


class RealUserProfile(BaseModel):
    user_id: int
    age: int
    gender: str
    income: int
    interaction_history: List[int]
    interaction_stats: Dict[str, int]
    interaction_pattern: str
    summary: str


class RealUsersResponse(BaseModel):
    users: List[RealUserProfile]
    total_count: int
    dataset_summary: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engines and real user selector on startup."""
    global recommendation_engine, enhanced_recommendation_engine, real_user_selector
    
    print("🚀 Starting 2-Phase Training API...")
    print("   Training approach: Pre-trained item tower + Joint fine-tuning")
    
    try:
        print("Loading 2-phase trained recommendation engine...")
        recommendation_engine = RecommendationEngine()
        print("✅ 2-phase recommendation engine loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading recommendation engine: {e}")
        recommendation_engine = None
    
    try:
        print("Loading enhanced recommendation engine...")
        # Try enhanced 128D engine first, fallback to regular enhanced
        try:
            from src.inference.enhanced_recommendation_engine_128d import Enhanced128DRecommendationEngine
            enhanced_recommendation_engine = Enhanced128DRecommendationEngine()
            print("✅ Using Enhanced 128D Recommendation Engine")
        except:
            from src.inference.enhanced_recommendation_engine import EnhancedRecommendationEngine
            enhanced_recommendation_engine = EnhancedRecommendationEngine()
            print("⚠️  Using fallback Enhanced Recommendation Engine")
        print("Enhanced recommendation engine loaded successfully!")
    except Exception as e:
        print(f"Error loading enhanced recommendation engine: {e}")
        enhanced_recommendation_engine = None
    
    try:
        print("Loading real user selector...")
        real_user_selector = RealUserSelector()
        print("Real user selector loaded successfully!")
    except Exception as e:
        print(f"Error loading real user selector: {e}")
        real_user_selector = None
    
    print("🎯 2-Phase API ready to serve recommendations!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Two-Tower Recommendation API (2-Phase Training)",
        "version": "1.0.0-2phase",
        "training_approach": "2-phase (pre-trained item tower + joint fine-tuning)",
        "status": "active" if recommendation_engine is not None else "initialization_failed"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if recommendation_engine is not None else "unhealthy",
        "engine_loaded": recommendation_engine is not None,
        "training_approach": "2-phase"
    }


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    return {
        "training_approach": "2-phase",
        "description": "Pre-trained item tower followed by joint training with user tower",
        "phases": [
            "Phase 1: Item tower pre-training on item features only",
            "Phase 2: Joint training of user tower + fine-tuning pre-trained item tower"
        ],
        "embedding_dimension": 128,
        "item_vocab_size": len(recommendation_engine.data_processor.item_vocab) if recommendation_engine.data_processor else "unknown",
        "artifacts_loaded": {
            "item_tower_pretrained": "src/artifacts/item_tower_weights",
            "item_tower_finetuned": "src/artifacts/item_tower_weights_finetuned_best",
            "user_tower": "src/artifacts/user_tower_weights_best",
            "rating_model": "src/artifacts/rating_model_weights_best"
        }
    }


@app.get("/real-users", response_model=RealUsersResponse)
async def get_real_users(count: int = 100, min_interactions: int = 5):
    """Get real user profiles with genuine interaction histories."""
    
    if real_user_selector is None:
        raise HTTPException(status_code=503, detail="Real user selector not available")
    
    try:
        # Get real user profiles
        real_users = real_user_selector.get_real_users(n=count, min_interactions=min_interactions)
        
        # Get dataset summary
        dataset_summary = real_user_selector.get_dataset_summary()
        
        # Format users for response
        formatted_users = []
        for user in real_users:
            formatted_users.append(RealUserProfile(**user))
        
        return RealUsersResponse(
            users=formatted_users,
            total_count=len(formatted_users),
            dataset_summary=dataset_summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving real users: {str(e)}")


@app.get("/real-users/{user_id}")
async def get_real_user_details(user_id: int):
    """Get detailed interaction breakdown for a specific real user."""
    
    if real_user_selector is None:
        raise HTTPException(status_code=503, detail="Real user selector not available")
    
    try:
        user_details = real_user_selector.get_user_interaction_details(user_id)
        
        if "error" in user_details:
            raise HTTPException(status_code=404, detail=user_details["error"])
        
        return user_details
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user details: {str(e)}")


@app.get("/dataset-summary")
async def get_dataset_summary():
    """Get summary statistics of the real dataset."""
    
    if real_user_selector is None:
        raise HTTPException(status_code=503, detail="Real user selector not available")
    
    try:
        return real_user_selector.get_dataset_summary()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset summary: {str(e)}")


@app.post("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get item recommendations for a user."""
    
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        user_profile = request.user_profile
        
        # Generate recommendations based on type
        if request.recommendation_type == "collaborative":
            recommendations = recommendation_engine.recommend_items_collaborative(
                age=user_profile.age,
                gender=user_profile.gender,
                income=user_profile.income,
                interaction_history=user_profile.interaction_history,
                k=request.num_recommendations
            )
        
        elif request.recommendation_type == "content":
            if not user_profile.interaction_history:
                raise HTTPException(
                    status_code=400, 
                    detail="Content-based recommendations require interaction history"
                )
            
            # Use most recent interaction as seed
            seed_item = user_profile.interaction_history[-1]
            recommendations = recommendation_engine.recommend_items_content_based(
                seed_item_id=seed_item,
                k=request.num_recommendations
            )
        
        elif request.recommendation_type == "hybrid":
            recommendations = recommendation_engine.recommend_items_hybrid(
                age=user_profile.age,
                gender=user_profile.gender,
                income=user_profile.income,
                interaction_history=user_profile.interaction_history,
                k=request.num_recommendations,
                collaborative_weight=request.collaborative_weight
            )
        
        elif request.recommendation_type == "enhanced":
            if enhanced_recommendation_engine is None:
                raise HTTPException(status_code=503, detail="Enhanced recommendation engine not available")
            
            # Check if it's the 128D engine or fallback
            if hasattr(enhanced_recommendation_engine, 'recommend_items_enhanced'):
                # 128D Enhanced engine
                recommendations = enhanced_recommendation_engine.recommend_items_enhanced(
                    age=user_profile.age,
                    gender=user_profile.gender,
                    income=user_profile.income,
                    interaction_history=user_profile.interaction_history,
                    k=request.num_recommendations,
                    diversity_weight=0.3 if request.enable_diversity else 0.0,
                    category_boost=request.category_boost if request.enable_category_boost else 1.0
                )
            else:
                # Fallback enhanced engine
                recommendations = enhanced_recommendation_engine.recommend_items_enhanced_hybrid(
                    age=user_profile.age,
                    gender=user_profile.gender,
                    income=user_profile.income,
                    interaction_history=user_profile.interaction_history,
                    k=request.num_recommendations,
                    collaborative_weight=request.collaborative_weight,
                    category_boost=request.category_boost,
                    enable_category_boost=request.enable_category_boost,
                    enable_diversity=request.enable_diversity
                )
        
        elif request.recommendation_type == "enhanced_128d":
            if enhanced_recommendation_engine is None or not hasattr(enhanced_recommendation_engine, 'recommend_items_enhanced'):
                raise HTTPException(status_code=503, detail="Enhanced 128D recommendation engine not available")
            
            recommendations = enhanced_recommendation_engine.recommend_items_enhanced(
                age=user_profile.age,
                gender=user_profile.gender,
                income=user_profile.income,
                interaction_history=user_profile.interaction_history,
                k=request.num_recommendations,
                diversity_weight=0.3 if request.enable_diversity else 0.0,
                category_boost=request.category_boost if request.enable_category_boost else 1.0
            )
        
        elif request.recommendation_type == "category_focused":
            if enhanced_recommendation_engine is None:
                raise HTTPException(status_code=503, detail="Enhanced recommendation engine not available")
            
            recommendations = enhanced_recommendation_engine.recommend_items_category_focused(
                age=user_profile.age,
                gender=user_profile.gender,
                income=user_profile.income,
                interaction_history=user_profile.interaction_history,
                k=request.num_recommendations,
                focus_percentage=0.8
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid recommendation_type. Must be 'collaborative', 'content', 'hybrid', 'enhanced', 'enhanced_128d', or 'category_focused'"
            )
        
        # Format response
        formatted_recommendations = []
        for item_id, score, item_info in recommendations:
            formatted_recommendations.append(
                RecommendationResponse(
                    item_id=item_id,
                    score=score,
                    item_info=ItemInfo(**item_info)
                )
            )
        
        return RecommendationsResponse(
            recommendations=formatted_recommendations,
            user_profile=user_profile,
            recommendation_type=request.recommendation_type,
            total_count=len(formatted_recommendations),
            training_approach="2-phase"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.post("/item-similarity", response_model=List[RecommendationResponse])
async def get_similar_items(request: ItemSimilarityRequest):
    """Get items similar to a given item."""
    
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        recommendations = recommendation_engine.recommend_items_content_based(
            seed_item_id=request.item_id,
            k=request.num_recommendations
        )
        
        formatted_recommendations = []
        for item_id, score, item_info in recommendations:
            formatted_recommendations.append(
                RecommendationResponse(
                    item_id=item_id,
                    score=score,
                    item_info=ItemInfo(**item_info)
                )
            )
        
        return formatted_recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")


@app.post("/predict-rating", response_model=RatingPredictionResponse)
async def predict_user_item_rating(request: RatingPredictionRequest):
    """Predict rating for a user-item pair."""
    
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        user_profile = request.user_profile
        
        predicted_rating = recommendation_engine.predict_rating(
            age=user_profile.age,
            gender=user_profile.gender,
            income=user_profile.income,
            interaction_history=user_profile.interaction_history,
            item_id=request.item_id
        )
        
        item_info = recommendation_engine._get_item_info(request.item_id)
        
        return RatingPredictionResponse(
            user_profile=user_profile,
            item_id=request.item_id,
            predicted_rating=predicted_rating,
            item_info=ItemInfo(**item_info)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")


@app.get("/items/{item_id}", response_model=ItemInfo)
async def get_item_info(item_id: int):
    """Get information about a specific item."""
    
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        item_info = recommendation_engine._get_item_info(item_id)
        return ItemInfo(**item_info)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving item info: {str(e)}")


@app.get("/items")
async def get_sample_items(limit: int = 20):
    """Get a sample of items for testing."""
    
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not available")
    
    try:
        # Get sample items from the dataframe
        sample_items = recommendation_engine.items_df.sample(n=min(limit, len(recommendation_engine.items_df)))
        
        items = []
        for _, row in sample_items.iterrows():
            items.append({
                "product_id": int(row['product_id']),
                "category_id": int(row['category_id']),
                "category_code": str(row['category_code']),
                "brand": str(row['brand']) if pd.notna(row['brand']) else 'Unknown',
                "price": float(row['price'])
            })
        
        return {"items": items, "total": len(items), "training_approach": "2-phase"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sample items: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting 2-Phase Training Recommendation API...")
    print("📊 Training approach: Pre-trained item tower + Joint fine-tuning")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_2phase:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )