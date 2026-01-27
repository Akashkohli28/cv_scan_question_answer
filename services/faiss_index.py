"""
FAISS index service for efficient semantic search over embeddings.
Maintains vector index with candidate_id metadata for filtered retrieval.
"""

import os
import json
from typing import Optional, List, Dict, Any
import numpy as np
import faiss
import logging

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Manage FAISS vector index for semantic search on CV embeddings."""
    
    def __init__(self, index_path: str = 'data/faiss.index'):
        """
        Initialize or load FAISS index.
        
        Args:
            index_path (str): Path to the FAISS index file
        """
        self.index_path = index_path
        self.metadata_path = index_path.replace('.index', '_metadata.json')
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: Dict[int, Dict] = {}
        self.next_id = 0
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create new index with L2 distance (Euclidean)
                # Dimension should match embedding dimension (1536 for text-embedding-3-small)
                self.index = faiss.IndexFlatL2(1536)
                logger.info("Created new FAISS index")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.next_id = max(int(k) for k in self.metadata.keys()) + 1 if self.metadata else 0
                logger.info(f"Loaded metadata with {len(self.metadata)} entries")
            else:
                self.metadata = {}
                self.next_id = 0
        
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Create fresh index
            self.index = faiss.IndexFlatL2(1536)
            self.metadata = {}
            self.next_id = 0
    
    def add_vector(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add a single embedding vector to the index.
        
        Args:
            embedding (np.ndarray): Embedding vector
            metadata (Dict): Metadata to associate with this vector (candidate_id, chunk_type, etc.)
                            Should include 'text' field with actual chunk content
            
        Returns:
            int: Index ID of the added vector
        """
        try:
            # Ensure embedding is 2D array
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Add to index
            self.index.add(embedding.astype(np.float32))
            
            # Store metadata - include the text content for direct retrieval
            vector_id = self.next_id
            self.metadata[str(vector_id)] = metadata
            self.next_id += 1
            
            logger.debug(f"Added vector {vector_id} for candidate {metadata.get('candidate_id')}")
            return vector_id
        
        except Exception as e:
            logger.error(f"Error adding vector: {str(e)}")
            raise
    
    def add_batch(self, embeddings: List[np.ndarray], metadatas: List[Dict]) -> List[int]:
        """
        Add multiple embedding vectors to the index.
        
        Args:
            embeddings (List[np.ndarray]): List of embedding vectors
            metadatas (List[Dict]): List of metadata dicts
            
        Returns:
            List[int]: List of vector IDs
        """
        try:
            # Stack embeddings into 2D array
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store metadata
            vector_ids = []
            for metadata in metadatas:
                vector_id = self.next_id
                self.metadata[str(vector_id)] = metadata
                vector_ids.append(vector_id)
                self.next_id += 1
            
            logger.info(f"Added batch of {len(vector_ids)} vectors")
            return vector_ids
        
        except Exception as e:
            logger.error(f"Error adding batch: {str(e)}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5,
               candidate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors in the index.
        
        Args:
            query_vector (np.ndarray): Query embedding
            k (int): Number of results to return
            candidate_id (Optional[str]): Filter results by candidate_id
            
        Returns:
            List[Dict]: List of search results with metadata and distances
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Reshape query vector
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(
                query_vector.astype(np.float32),
                min(k, self.index.ntotal)
            )
            
            # Filter and format results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                metadata = self.metadata.get(str(idx), {})
                
                # Filter by candidate_id if specified
                if candidate_id and metadata.get('candidate_id') != candidate_id:
                    continue
                
                results.append({
                    'vector_id': int(idx),
                    'distance': float(dist),
                    'metadata': metadata
                })
            
            # Truncate to k results after filtering
            results = results[:k]
            
            logger.debug(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            return []
    
    def get_candidate_chunks(self, candidate_id: str) -> List[Dict[str, Any]]:
        """
        Get all indexed chunks for a specific candidate.
        
        Args:
            candidate_id (str): The candidate's unique identifier
            
        Returns:
            List[Dict]: List of chunks with their metadata
        """
        try:
            chunks = []
            for vector_id, metadata in self.metadata.items():
                if metadata.get('candidate_id') == candidate_id:
                    chunks.append({
                        'vector_id': vector_id,
                        'chunk_type': metadata.get('chunk_type'),
                        'section': metadata.get('section'),
                        'metadata': metadata
                    })
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error retrieving candidate chunks: {str(e)}")
            return []
    
    def save(self):
        """Save FAISS index and metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"Saved FAISS index and metadata to {self.index_path}")
        
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def delete_candidate(self, candidate_id: str):
        """
        Remove all vectors associated with a candidate.
        Note: FAISS doesn't support deletion, so we mark as removed in metadata.
        
        Args:
            candidate_id (str): The candidate's unique identifier
        """
        try:
            removed_count = 0
            keys_to_remove = []
            
            for vector_id, metadata in self.metadata.items():
                if metadata.get('candidate_id') == candidate_id:
                    metadata['removed'] = True
                    keys_to_remove.append(vector_id)
                    removed_count += 1
            
            logger.info(f"Marked {removed_count} vectors for candidate {candidate_id} as removed")
        
        except Exception as e:
            logger.error(f"Error deleting candidate: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dict: Index statistics
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.index.d if self.index else 0,
            'metadata_entries': len(self.metadata),
            'index_path': self.index_path,
            'metadata_path': self.metadata_path
        }
