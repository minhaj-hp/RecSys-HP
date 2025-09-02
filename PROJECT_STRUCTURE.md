# Project Structure

This recommendation system is organized into specialized modules for maximum maintainability and extensibility:

```
RecSys-HP/
│
├── 🧠 src/                               # Core machine learning pipeline
│   ├── data_generation/                  # Data synthesis and augmentation
│   │   └── generate_demographics.py      # Synthetic user demographics
│   │
│   ├── preprocessing/                    # Data preparation pipeline
│   │   ├── data_loader.py               # Dataset loading and validation
│   │   └── user_data_preparation.py     # User feature engineering with categorization
│   │
│   ├── models/                           # Neural network architectures
│   │   ├── item_tower.py                # Enhanced item embedding tower
│   │   ├── item_tower_original.py       # Original item tower (deprecated)
│   │   └── user_tower.py                # User tower with categorical demographics
│   │
│   ├── training/                         # Training pipelines
│   │   ├── item_pretraining.py          # Phase 1: Item tower pre-training
│   │   └── joint_training.py            # Joint training of both towers
│   │
│   ├── inference/                        # Production serving components
│   │   ├── faiss_index.py               # Vector similarity search
│   │   ├── recommendation_engine.py     # Core inference pipeline
│   │   └── demographic_clustering.py    # User demographic analysis
│   │
│   ├── utils/                            # Utility functions
│   │   └── real_user_selector.py        # Real user data selection for testing
│   │
│   └── artifacts/                        # Model checkpoints and metadata
│       ├── *.data-* / *.index           # TensorFlow model weights
│       ├── *.npy                        # Numpy arrays (embeddings)
│       ├── *.pkl                        # Pickled features/vocabularies
│       └── *.bin                        # FAISS indices
│
├── 🚀 scripts/                           # Training and utility scripts
│   ├── run_training_pipeline.py         # Main training orchestration
│   ├── run_2phase_training.py           # 2-phase training approach
│   ├── run_joint_training.py            # Joint training approach
│   └── visualize_embeddings.py          # Embedding visualization
│
├── 🌐 api/                               # FastAPI backend
│   └── main.py                          # Production API server
│
├── 💻 frontend/                          # React web interface
│   ├── src/                             # React components
│   │   ├── App.js                       # Main application
│   │   └── *.css                        # Styling
│   ├── public/                          # Static assets
│   ├── build/                           # Production build
│   └── package.json                     # Node.js dependencies
│
├── 🧪 tests/                             # Comprehensive test suite
│   ├── analysis/                        # Analysis and evaluation scripts
│   ├── test_category_boosted_recommendations.py # Category boosted testing
│   └── TEST_USAGE.md                    # Testing documentation
│
├── 📊 datasets/                          # Training data
│   ├── users.csv                        # User demographics and profiles
│   ├── interactions.csv                 # User-item interaction data
│   └── items.csv                        # Item features and metadata
│
└── 📋 Configuration Files
    ├── requirements.txt                 # Python dependencies
    ├── README.md                        # This documentation
    ├── ARCHITECTURE.md                  # Detailed architecture guide
    └── DEEP_ARCHITECTURE.md             # Deep technical specifications
```

## Key Components

### Core Engine (`src/`)
- **Models**: Enhanced two-tower architecture with 128D embeddings
- **Training**: Multi-phase training pipeline with curriculum learning
- **Inference**: Real-time recommendation serving with FAISS indexing
- **Preprocessing**: Data preparation with categorical demographics

### API & Frontend (`api/`, `frontend/`)
- **FastAPI Backend**: Production-ready API with comprehensive error handling
- **React Frontend**: Interactive web interface with real-time recommendations

### Training Scripts (`scripts/`)
- **Pipeline Training**: Complete end-to-end training orchestration
- **Multi-Phase Training**: Separate item pretraining and joint optimization
- **Visualization**: Embedding analysis and quality assessment

### Testing & Analysis (`tests/`)
- **Comprehensive Testing**: Quality assurance and performance validation
- **Analysis Tools**: Recommendation quality and category alignment evaluation