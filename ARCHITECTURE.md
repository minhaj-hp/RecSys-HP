# RecSys-HP Architecture Documentation

## System Overview

RecSys-HP is a comprehensive recommendation system built with a **Two-Tower Architecture** that provides real-time personalized recommendations through a web interface. The system combines machine learning models, real-time inference, and an interactive frontend.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RecSys-HP System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │                 │    │                 │    │                 │        │
│  │    Frontend     │◄──►│   FastAPI       │◄──►│   ML Pipeline   │        │
│  │   (React.js)    │    │   Backend       │    │  (TensorFlow)   │        │
│  │                 │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                │                          │                 │
│                                │                          │                 │
│                          ┌─────▼─────┐            ┌─────▼─────┐           │
│                          │           │            │           │           │
│                          │  Real     │            │  Trained  │           │
│                          │  Dataset  │            │  Models   │           │
│                          │  (CSV)    │            │ (Weights) │           │
│                          │           │            │           │           │
│                          └───────────┘            └───────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. Frontend Layer (React.js)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                  │
│  │   Real User   │  │  Custom User  │  │ Recommendations│                  │
│  │   Interface   │  │   Interface   │  │   Display      │                  │
│  └───────────────┘  └───────────────┘  └───────────────┘                  │
│           │                   │                   │                        │
│           └───────────────────┼───────────────────┘                        │
│                               │                                            │
│  ┌─────────────────────────────▼─────────────────────────────┐             │
│  │                 App.js (Main Component)                   │             │
│  │                                                           │             │
│  │  • State Management (useState, useMemo)                  │             │
│  │  • API Communication (axios)                             │             │
│  │  • User Profile Management                               │             │
│  │  • Interaction History Tracking                          │             │
│  │  • Category Analysis & Visualization                     │             │
│  │  • Recommendation Engine Interface                       │             │
│  └───────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Frontend Features:**
- **Real User Selection**: Browse and select from actual user profiles
- **Custom User Creation**: Define synthetic user profiles with behavioral patterns
- **Category Analysis**: Real-time category percentage calculations
- **Recommendation Display**: Paginated, categorized recommendations
- **Interactive Timeline**: User interaction history with item details
- **Similar Items Modal**: Click any recommendation to see category-balanced similar items (60% same category, 40% discovery)
- **ANN-Powered Search**: FAISS similarity scores with visual similarity indicators

### 2. Backend API Layer (FastAPI)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   API Routes    │  │  Data Services  │  │   ML Interface  │           │
│  │                 │  │                 │  │                 │           │
│  │ /real-users     │  │ RealUserSelect  │  │ Recommendation  │           │
│  │ /behavioral-    │  │                 │  │ Engine          │           │
│  │  patterns       │  │ DataProcessor   │  │                 │           │
│  │ /recommendations│  │                 │  │ Enhanced Engine │           │
│  │ /items          │  │ Dataset Loaders │  │                 │           │
│  │ /predict-rating │  │                 │  │ FAISS Index     │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│           │                        │                        │               │
│           └────────────────────────┼────────────────────────┘               │
│                                    │                                        │
│  ┌─────────────────────────────────▼─────────────────────────────┐         │
│  │                      main.py                                  │         │
│  │                                                               │         │
│  │  • FastAPI Application Setup                                 │         │
│  │  • CORS Configuration                                        │         │
│  │  • Pydantic Models for Request/Response                      │         │
│  │  • Global Instance Management                                │         │
│  │  • Error Handling & HTTP Status Codes                       │         │
│  │  • Category Filtering & Data Processing                      │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**API Endpoints:**
- `GET /real-users` - Fetch real user profiles with interaction stats
- `GET /real-users/{user_id}` - Get detailed user interaction timeline
- `GET /behavioral-patterns` - Get enriched behavioral patterns with item details
- `POST /recommendations` - Generate personalized recommendations
- `POST /item-similarity` - Get similar items with category constraints (60% same category)
- `GET /items/{item_id}` - Get individual item information
- `POST /predict-rating` - Predict user-item rating

### 3. Machine Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML Pipeline Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    Two-Tower Architecture                             │ │
│  │                                                                       │ │
│  │  ┌─────────────┐                                ┌─────────────┐      │ │
│  │  │             │                                │             │      │ │
│  │  │ User Tower  │                                │ Item Tower  │      │ │
│  │  │             │                                │             │      │ │
│  │  │ • Age       │                                │ • Product   │      │ │
│  │  │ • Gender    │                                │ • Category  │      │ │
│  │  │ • Income    │                                │ • Brand     │      │ │
│  │  │ • History   │                                │ • Price     │      │ │
│  │  │             │                                │             │      │ │
│  │  └──────┬──────┘                                └──────┬──────┘      │ │
│  │         │                                              │             │ │
│  │         ▼                                              ▼             │ │
│  │  ┌─────────────┐                                ┌─────────────┐      │ │
│  │  │ User Embed  │                                │ Item Embed  │      │ │
│  │  │ (128D/64D)  │                                │ (128D/64D)  │      │ │
│  │  └──────┬──────┘                                └──────┬──────┘      │ │
│  │         │                                              │             │ │
│  │         └──────────────────┐    ┌──────────────────────┘             │ │
│  │                            ▼    ▼                                    │ │
│  │                      ┌─────────────┐                                 │ │
│  │                      │ Dot Product │                                 │ │
│  │                      │   Scoring   │                                 │ │
│  │                      └─────────────┘                                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Inference Engines                                │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │ │
│  │  │ Basic Engine    │  │ Enhanced Engine │  │ 128D Engine     │      │ │
│  │  │                 │  │                 │  │                 │      │ │
│  │  │ • Collaborative │  │ • Hybrid Rec    │  │ • Advanced      │      │ │
│  │  │ • Content-Based │  │ • Category      │  │ • Diversity     │      │ │
│  │  │ • Hybrid        │  │   Boosting      │  │ • Category      │      │ │
│  │  │                 │  │ • Diversity     │  │   Focus         │      │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       FAISS Index                                    │ │
│  │                                                                       │ │
│  │  • Fast Similarity Search                                            │ │
│  │  • Item Embeddings Storage                                           │ │
│  │  • Efficient Nearest Neighbor Retrieval                             │ │
│  │  • Metadata Mapping (item_id → embedding)                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Data Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │  Raw Datasets   │  │  Processed      │  │  Model          │           │
│  │   (datasets/)   │  │   Features      │  │ Artifacts       │           │
│  │                 │  │ (src/artifacts/)│  │(src/artifacts/) │           │
│  │ • users.csv     │  │                 │  │                 │           │
│  │ • items.csv     │  │ • vocabularies  │  │ • model weights │           │
│  │ • interactions  │  │ • embeddings    │  │ • FAISS index   │           │
│  │   .csv          │  │ • features.pkl  │  │ • config files  │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### 1. User Selection & Profile Loading Flow

```
Frontend                    Backend                    Data Layer
   │                          │                          │
   │──── GET /real-users ────►│                          │
   │                          │── Load user profiles ──►│
   │                          │                          │
   │◄─── User profiles ───────│◄─── users.csv ──────────│
   │                          │                          │
   │── Select user ───────────│                          │
   │                          │                          │
   │─ GET /real-users/{id} ──►│                          │
   │                          │─── Join interactions ──►│
   │                          │─── with items data ────►│
   │                          │                          │
   │◄─ Enriched timeline ─────│◄── Combined data ───────│
   │                          │                          │
   │── Calculate categories ──│                          │
   │── Display profile ───────│                          │
```

### 2. Behavioral Pattern Enhancement Flow

```
Frontend                    Backend                    ML Engine
   │                          │                          │
   │─ GET /behavioral-────────│                          │
   │   patterns              │                          │
   │                          │── Get real users ──────►│
   │                          │                          │
   │                          │◄── User profiles ───────│
   │                          │                          │
   │                          │── For each item_id: ────│
   │                          │   get_item_info() ─────►│
   │                          │                          │
   │                          │◄── Item details ────────│
   │                          │                          │
   │◄── Enriched patterns ────│── Combine data ─────────│
   │                          │                          │
   │── Store enriched ────────│                          │
   │   interactions          │                          │
   │── Calculate categories ──│                          │
   │── Display immediately ───│                          │
```

### 3. Recommendation Generation Flow

```
Frontend                    Backend                    ML Engine
   │                          │                          │
   │── POST /recommendations ►│                          │
   │   {user_profile,         │                          │
   │    recommendation_type,  │                          │
   │    category_filter}      │                          │
   │                          │                          │
   │                          │── Prepare features ────►│
   │                          │   (age, gender,          │
   │                          │    income, history)      │
   │                          │                          │
   │                          │                          │──┐
   │                          │                          │  │ Choose Engine:
   │                          │                          │  │ • collaborative
   │                          │                          │  │ • content
   │                          │                          │  │ • hybrid
   │                          │                          │  │ • enhanced
   │                          │                          │  │ • enhanced_128d
   │                          │                          │──┘
   │                          │                          │
   │                          │◄── Recommendations ─────│
   │                          │   [item_id, score,       │
   │                          │    item_info]            │
   │                          │                          │
   │                          │── Apply category ───────│
   │                          │   filtering              │
   │                          │                          │
   │◄── Filtered results ─────│                          │
   │                          │                          │
   │── Display paginated ─────│                          │
   │   recommendations       │                          │
```

### 4. Category Analysis Flow

```
User Interactions           Category Processor          Display
      │                           │                        │
      │── Select interactions ───►│                        │
      │                           │                        │
      │                           │──┐                     │
      │                           │  │ Process by type:    │
      │                           │  │ • Real user        │
      │                           │  │ • Behavioral       │
      │                           │  │ • Synthetic        │
      │                           │──┘                     │
      │                           │                        │
      │                           │── Extract categories ──│
      │                           │── Calculate % ─────────│
      │                           │                        │
      │                           │                        │──► User Interests
      │                           │                        │
      │── Get recommendations ────│                        │
      │                           │                        │
      │                           │── Process rec cats ────│
      │                           │── Calculate % ─────────│
      │                           │                        │
      │                           │                        │──► Recommendation
      │                           │                        │    Categories
      │                           │                        │
      │                           │── Compare & highlight ─│
      │                           │   matching categories  │
      │                           │                        │──► Category
      │                           │                        │    Matching
```

### 5. Similar Items with Category Constraints Flow

```
Frontend                    Backend                    FAISS/ML Engine
   │                          │                          │
   │── Click recommendation ──│                          │
   │                          │                          │
   │── POST /item-similarity ►│                          │
   │   {item_id: 1004565}     │                          │
   │                          │                          │
   │                          │── Get seed category ───►│
   │                          │   (electronics.smartphone) │
   │                          │                          │
   │                          │── FAISS search (k*3) ──►│
   │                          │   Get 30 candidates      │
   │                          │                          │
   │                          │◄─── Similar items ──────│
   │                          │   [30 items + scores]    │
   │                          │                          │
   │                          │──┐                       │
   │                          │  │ Category Separation:  │
   │                          │  │ • Same: smartphones   │
   │                          │  │ • Different: tablets, │
   │                          │  │   laptops, etc.       │
   │                          │──┘                       │
   │                          │                          │
   │                          │──┐                       │
   │                          │  │ Apply 60/40 Rule:     │
   │                          │  │ • 6 smartphones (60%) │
   │                          │  │ • 4 others (40%)      │
   │                          │──┘                       │
   │                          │                          │
   │◄── Category-balanced ────│                          │
   │   similar items         │                          │
   │   [10 items, 60% same   │                          │
   │    category]            │                          │
   │                          │                          │
   │── Display modal with ────│                          │
   │   similarity scores     │                          │
   │   & category distribution│                          │
```

## File Structure & Responsibilities

```
RecSys-HP/
├── frontend/
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # Styling
│   │   └── index.js            # React entry point
│   └── package.json            # Frontend dependencies
│
├── api/
│   └── main.py                 # FastAPI application
│
├── src/
│   ├── inference/              # Recommendation engines
│   │   ├── recommendation_engine.py          # Basic engine
│   │   ├── enhanced_recommendation_engine.py # Enhanced engine
│   │   └── enhanced_recommendation_engine_128d.py # 128D engine
│   │
│   ├── models/                 # Neural network models
│   │   ├── user_tower.py       # User embedding model
│   │   ├── item_tower.py       # Item embedding model
│   │   └── enhanced_two_tower.py # Enhanced architecture
│   │
│   ├── preprocessing/          # Data processing
│   │   └── data_loader.py      # DataProcessor class
│   │
│   ├── training/               # Model training
│   │   ├── joint_training.py   # Main training loop
│   │   └── improved_joint_training.py # Enhanced training
│   │
│   ├── utils/                  # Utility functions
│   │   └── real_user_selector.py # RealUserSelector class
│   │
│   └── artifacts/              # Trained models & data
│       ├── vocabularies.pkl    # Feature vocabularies
│       ├── *_weights.*         # Model weights
│       └── faiss_*             # FAISS index files
│
├── datasets/                   # Raw data
│   ├── users.csv              # User demographics
│   ├── items.csv              # Product catalog
│   └── interactions.csv        # User-item interactions
│
└── training scripts            # Training orchestration
    ├── run_training_pipeline.py
    └── train_improved_model.py
```

## Key Classes & Functions

### Frontend (App.js)
- **State Management**: `useState` hooks for UI state
- **Data Fetching**: `fetchRealUsers()`, `fetchRandomBehavioralPatterns()`
- **Category Analysis**: `getCategoryPercentages()`, `getRecommendationCategoryPercentages()`
- **User Interactions**: `handleRealUserSelect()`, `generateRealisticInteractions()`

### Backend (main.py)
- **API Endpoints**: RESTful routes with Pydantic models
- **Data Integration**: Combines ML engine with real user data
- **Category Filtering**: `filter_interactions_by_category()`, `filter_recommendations_by_category()`

### ML Inference (recommendation_engine.py)
- **Recommendation Methods**: `recommend_items_collaborative()`, `recommend_items_content_based()`, `recommend_items_hybrid()`
- **Category-Aware Similarity**: Enhanced `recommend_items_content_based()` with optional category constraints
- **Model Loading**: Loads trained TensorFlow models and FAISS indices
- **Feature Processing**: Converts user/item data to model inputs
- **FAISS Integration**: Fast ANN search with category-balanced results (60/40 split)

### Data Processing (data_loader.py)
- **DataProcessor**: Handles vocabulary building and feature preparation
- **Dataset Creation**: `create_positive_negative_pairs()`, `create_tf_dataset()`

### User Selection (real_user_selector.py)
- **RealUserSelector**: Extracts genuine user profiles from datasets
- **Interaction Analysis**: `get_user_interaction_details()`, categorizes user patterns

## Performance Considerations

### Frontend Optimizations
- **useMemo**: Prevents unnecessary re-calculations
- **Pagination**: Limits displayed recommendations
- **Debounced API calls**: Reduces server load

### Backend Optimizations
- **FAISS Indexing**: Fast similarity search for item recommendations
- **Batch Processing**: Efficient data processing in parallel
- **Caching**: Global instance management for models

### ML Optimizations
- **Two-Tower Architecture**: Separates user and item processing
- **Embedding Caching**: Pre-computed item embeddings
- **TensorFlow Optimization**: GPU acceleration when available

## Security & Error Handling

### API Security
- **CORS Configuration**: Cross-origin request handling
- **Input Validation**: Pydantic models validate all inputs
- **Error Responses**: Structured HTTP error codes

### Error Handling
- **Try-Catch Blocks**: Comprehensive error catching
- **Graceful Degradation**: Fallback to simpler models when enhanced unavailable
- **User Feedback**: Clear error messages in UI

This architecture provides a scalable, maintainable recommendation system with real-time inference capabilities and comprehensive user interaction analysis.