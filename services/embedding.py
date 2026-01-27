"""
Embedding service for generating vector embeddings from text.
Uses OpenAI's embedding API to create dense vectors for semantic search.
"""

import os
from typing import List, Optional
import numpy as np
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate and manage text embeddings using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding service.
        
        Args:
            api_key (Optional[str]): OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model (str): OpenAI embedding model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_dim = 1536  # For text-embedding-3-small
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Sort by index to ensure order
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [np.array(e.embedding, dtype=np.float32) for e in embeddings]
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Euclidean distance
        """
        return float(np.linalg.norm(embedding1 - embedding2))
