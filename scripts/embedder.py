#!/usr/bin/env python3
"""
Embedder module - compatibility wrapper for vector memory system.
Provides backward compatibility for embed() function calls.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def _simple_embed(text: str) -> List[float]:
    """Simple fallback embedding using basic text features."""
    import hashlib
    
    # Create a deterministic but distributed embedding
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to float values in range [-1, 1]
    embedding = []
    for i in range(0, len(hash_bytes), 2):
        if i + 1 < len(hash_bytes):
            val = (hash_bytes[i] + hash_bytes[i+1] * 256) / 65535.0 * 2 - 1
            embedding.append(val)
    
    # Pad or truncate to 384 dimensions
    while len(embedding) < 384:
        embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
    
    return embedding[:384]

def embed(text: str) -> Optional[List[float]]:
    """
    Generate embeddings for text using simple fallback method.
    
    Args:
        text: Text to embed
        
    Returns:
        List of float values representing the embedding
    """
    try:
        if not text or not text.strip():
            return [0.0] * 384
            
        return _simple_embed(text)
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * 384  # Return zero vector on error

def get_embeddings_cached(text: str) -> Optional[List[float]]:
    """Alias for embed function for backward compatibility."""
    return embed(text)

def most_similar(query_embedding: List[float], candidates: List[List[float]], 
                top_k: int = 5) -> List[int]:
    """
    Find most similar embeddings using cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        candidates: List of candidate embedding vectors
        top_k: Number of top results to return
        
    Returns:
        List of indices of most similar candidates
    """
    try:
        if not query_embedding or not candidates:
            return []
        
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array(candidates)
        
        # Calculate cosine similarities
        similarities = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()
        
    except Exception as e:
        logger.error(f"Error in most_similar: {e}")
        return []

# Export the main functions
__all__ = ['embed', 'get_embeddings_cached', 'most_similar']