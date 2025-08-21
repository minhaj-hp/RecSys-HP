import faiss
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional


class FAISSItemIndex:
    """FAISS-based item similarity search index."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.index = None
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.item_embeddings = None
        
    def build_index(self, 
                   item_embeddings: Dict[int, np.ndarray],
                   index_type: str = "IVF") -> None:
        """Build FAISS index from item embeddings."""
        
        # Convert embeddings dict to arrays
        item_ids = list(item_embeddings.keys())
        embeddings_array = np.array(list(item_embeddings.values())).astype('float32')
        
        print(f"Building FAISS index for {len(item_ids)} items...")
        print(f"Embedding shape: {embeddings_array.shape}")
        
        # Create mappings
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_item_id = {idx: item_id for idx, item_id in enumerate(item_ids)}
        self.item_embeddings = embeddings_array
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Choose index type
        if index_type == "Flat":
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "IVF":
            # Approximate search (faster)
            nlist = min(100, len(item_ids) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            self.index.train(embeddings_array)
            self.index.nprobe = min(10, nlist)  # Search in top 10 clusters
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        print(f"FAISS index built successfully with {self.index.ntotal} items")
    
    def search_similar_items(self, 
                           query_item_id: int,
                           k: int = 10,
                           exclude_query: bool = True) -> List[Tuple[int, float]]:
        """Find k most similar items to query item."""
        
        if query_item_id not in self.item_id_to_idx:
            print(f"Item {query_item_id} not found in index")
            return []
        
        # Get query embedding
        query_idx = self.item_id_to_idx[query_item_id]
        query_embedding = self.item_embeddings[query_idx:query_idx+1]
        
        # Search
        search_k = k + 1 if exclude_query else k
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Convert results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
                
            item_id = self.idx_to_item_id[idx]
            
            # Skip the query item itself if requested
            if exclude_query and item_id == query_item_id:
                continue
                
            results.append((item_id, float(score)))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_embedding(self, 
                          query_embedding: np.ndarray,
                          k: int = 10) -> List[Tuple[int, float]]:
        """Find k most similar items to query embedding."""
        
        # Ensure correct shape and normalization
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item_id = self.idx_to_item_id[idx]
            results.append((item_id, float(score)))
        
        return results
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get embedding for a specific item."""
        if item_id not in self.item_id_to_idx:
            return None
        
        idx = self.item_id_to_idx[item_id]
        return self.item_embeddings[idx]
    
    def validate_index(self, sample_queries: List[int] = None) -> None:
        """Validate the index by running sample similarity searches."""
        
        if sample_queries is None:
            # Use first 5 items as sample queries
            sample_queries = list(self.item_id_to_idx.keys())[:5]
        
        print("Validating FAISS index...")
        
        for query_item in sample_queries:
            if query_item not in self.item_id_to_idx:
                continue
                
            similar_items = self.search_similar_items(query_item, k=5)
            
            print(f"\nSimilar items to {query_item}:")
            for item_id, score in similar_items:
                print(f"  Item {item_id}: similarity = {score:.4f}")
    
    def save_index(self, save_path: str = "src/artifacts/") -> None:
        """Save FAISS index and mappings."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}/faiss_item_index.bin")
        
        # Save mappings and metadata
        metadata = {
            'item_id_to_idx': self.item_id_to_idx,
            'idx_to_item_id': self.idx_to_item_id,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{save_path}/faiss_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save embeddings
        np.save(f"{save_path}/faiss_item_embeddings.npy", self.item_embeddings)
        
        print(f"FAISS index saved to {save_path}")
    
    def load_index(self, load_path: str = "src/artifacts/") -> None:
        """Load FAISS index and mappings."""
        
        # Load FAISS index
        self.index = faiss.read_index(f"{load_path}/faiss_item_index.bin")
        
        # Load metadata
        with open(f"{load_path}/faiss_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.item_id_to_idx = metadata['item_id_to_idx']
        self.idx_to_item_id = metadata['idx_to_item_id']
        self.embedding_dim = metadata['embedding_dim']
        
        # Load embeddings
        self.item_embeddings = np.load(f"{load_path}/faiss_item_embeddings.npy")
        
        print(f"FAISS index loaded from {load_path}")
        print(f"Index contains {self.index.ntotal} items")


def main():
    """Main function to build FAISS index from pre-trained embeddings."""
    
    # Load item embeddings
    print("Loading item embeddings...")
    item_embeddings = np.load("src/artifacts/item_embeddings.npy", allow_pickle=True).item()
    
    # Create and build FAISS index
    print("Building FAISS index...")
    faiss_index = FAISSItemIndex(embedding_dim=64)
    faiss_index.build_index(item_embeddings, index_type="IVF")
    
    # Validate index
    faiss_index.validate_index()
    
    # Save index
    print("Saving FAISS index...")
    faiss_index.save_index()
    
    print("FAISS index creation completed!")


if __name__ == "__main__":
    main()