from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import sys
import pandas as pd

# Add src to path for imports and set working directory
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
os.chdir(parent_dir)  # Change to project root directory

from src.inference.recommendation_engine import RecommendationEngine

# Initialize FastAPI app
app = FastAPI(
    title="Two-Tower Recommendation API",
    description="API for serving recommendations using a two-tower architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommendation engine instance
recommendation_engine = None


# Pydantic models for request/response
class UserProfile(BaseModel):
    age: int
    gender: str  # "male" or "female"
    income: float
    interaction_history: Optional[List[int]] = []


class RecommendationRequest(BaseModel):
    user_profile: UserProfile
    num_recommendations: int = 10
    recommendation_type: str = "hybrid"  # "collaborative", "content", "hybrid"
    collaborative_weight: Optional[float] = 0.7


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


class RatingPredictionResponse(BaseModel):
    user_profile: UserProfile
    item_id: int
    predicted_rating: float
    item_info: ItemInfo


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup."""
    global recommendation_engine
    
    try:
        print("Loading recommendation engine...")
        recommendation_engine = RecommendationEngine()
        print("Recommendation engine loaded successfully!")
    except Exception as e:
        print(f"Error loading recommendation engine: {e}")
        recommendation_engine = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Two-Tower Recommendation API",
        "version": "1.0.0",
        "status": "active" if recommendation_engine is not None else "initialization_failed"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if recommendation_engine is not None else "unhealthy",
        "engine_loaded": recommendation_engine is not None
    }


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
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid recommendation_type. Must be 'collaborative', 'content', or 'hybrid'"
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
            total_count=len(formatted_recommendations)
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
        
        return {"items": items, "total": len(items)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sample items: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )