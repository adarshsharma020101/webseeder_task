import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    """FAISS vector store for similarity search"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index
        
        Args:
            dimension: Dimension of embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Add embeddings and corresponding text chunks
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: List of text chunks
        """
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []