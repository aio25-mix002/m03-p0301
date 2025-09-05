import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
import hashlib
from sentence_transformers import SentenceTransformer

class FAISSCache:
    def __init__(self, cache_dir: str = "./cache/faiss"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.metadata = {}
        self.embedding_dim = None
        
    def _get_index_path(self, dataset_hash: str) -> Path:
        """Generate index file path"""
        return self.cache_dir / f"faiss_index_{dataset_hash}.index"
    
    def _get_metadata_path(self, dataset_hash: str) -> Path:
        """Generate metadata file path"""
        return self.cache_dir / f"metadata_{dataset_hash}.pkl"
    
    def _hash_dataset(self, texts: List[str], model_name: str) -> str:
        """Generate hash for dataset"""
        content = f"{model_name}_{len(texts)}_{hashlib.md5(''.join(texts).encode()).hexdigest()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def build_index(self, embeddings: np.ndarray, texts: List[str], 
                   model_name: str, index_type: str = "flat") -> str:
        """Build FAISS index and cache it"""
        dataset_hash = self._hash_dataset(texts, model_name)
        index_path = self._get_index_path(dataset_hash)
        metadata_path = self._get_metadata_path(dataset_hash)
        
        # Check if index already exists
        if index_path.exists() and metadata_path.exists():
            print(f"Loading existing FAISS index: {dataset_hash}")
            return self.load_index(dataset_hash)
        
        print(f"Building new FAISS index: {dataset_hash}")
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        self.embedding_dim = embedding_dim
        
        if index_type == "flat":
            # Exact search (slower but accurate)
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        elif index_type == "ivf":
            # Approximate search (faster but less accurate)
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            self.index.train(embeddings.astype('float32'))
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World (good balance)
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = {
            'texts': texts,
            'model_name': model_name,
            'embedding_dim': embedding_dim,
            'index_type': index_type,
            'dataset_hash': dataset_hash,
            'num_vectors': len(embeddings)
        }
        
        # Save index and metadata
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"FAISS index saved: {index_path}")
        return dataset_hash
    
    def load_index(self, dataset_hash: str) -> str:
        """Load existing FAISS index"""
        index_path = self._get_index_path(dataset_hash)
        metadata_path = self._get_metadata_path(dataset_hash)
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index not found: {dataset_hash}")
        
        # Load index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.embedding_dim = self.metadata['embedding_dim']
        print(f"FAISS index loaded: {dataset_hash}")
        return dataset_hash
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get corresponding texts
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata['texts']):
                results.append(self.metadata['texts'][idx])
        
        return scores[0], results
    
    def get_similar_documents(self, query_text: str, model: SentenceTransformer, 
                            k: int = 5) -> List[Tuple[str, float]]:
        """Get similar documents for a text query"""
        # Generate query embedding
        query_embedding = model.encode([query_text], normalize_embeddings=True)
        
        # Search
        scores, texts = self.search(query_embedding, k)
        
        # Return results with scores
        return list(zip(texts, scores))