# vector_memory/hybrid_memory.py

"""
Hybrid Vector Memory System for MeRNSTA

This module implements Phase 17: Hybrid Memory Intelligence, enabling
parallel queries across FAISS, HRRFormer, and VecSymR backends with
intelligent result fusion.
"""

import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cosine, euclidean

from .config import load_memory_config

@dataclass
class HybridMemoryResult:
    """Result from hybrid memory search with source attribution."""
    
    content: str
    vector: List[float]
    confidence: float
    recency_score: float
    semantic_overlap: float
    source_backend: str
    original_score: float
    hybrid_score: float
    metadata: Dict[str, Any]

@dataclass
class BackendResult:
    """Internal result from individual backend."""
    
    backend: str
    vector: List[float]
    success: bool
    error: Optional[str] = None
    latency: float = 0.0

class HybridVectorMemory:
    """
    Hybrid Vector Memory system that intelligently routes queries across
    multiple backends and fuses results using confidence weighting,
    recency, semantic overlap, and source scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid memory system.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or load_memory_config()
        self.hybrid_mode = self.config.get('hybrid_mode', False)
        self.hybrid_strategy = self.config.get('hybrid_strategy', 'ensemble')
        self.backends = self.config.get('hybrid_backends', ['default', 'hrrformer', 'vecsymr'])
        self.weights = self.config.get('backend_weights', {
            'default': 0.4,      # Semantic search baseline
            'hrrformer': 0.35,   # Symbolic reasoning
            'vecsymr': 0.25      # Analogical mapping
        })
        
        # Parallel execution settings
        self.max_workers = self.config.get('max_workers', 3)
        self.timeout = self.config.get('backend_timeout', 10.0)
        
        # Result fusion parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.semantic_weight = self.config.get('semantic_weight', 0.5)
        self.source_weight = self.config.get('source_weight', 0.3)
        
        # Initialize vectorizers
        self.vectorizers = {}
        self._initialize_vectorizers()
        
        logging.info(f"✅ HybridVectorMemory initialized with strategy: {self.hybrid_strategy}")
        logging.info(f"🔄 Active backends: {list(self.vectorizers.keys())}")
    
    def _initialize_vectorizers(self):
        """Initialize all available vectorizers."""
        from .hrrformer_adapter import hrrformer_vectorize
        from .vecsymr_adapter import vecsymr_vectorize
        
        vectorizer_map = {
            'hrrformer': hrrformer_vectorize,
            'vecsymr': vecsymr_vectorize
        }
        
        for backend in self.backends:
            try:
                if backend in vectorizer_map:
                    self.vectorizers[backend] = vectorizer_map[backend]
                elif backend == 'default':
                    # Import default vectorizer inline to avoid circular import
                    from scripts.embedder import embed
                    import numpy as np
                    def default_vectorize(text: str) -> list:
                        embedding = embed(text)
                        if isinstance(embedding, np.ndarray):
                            return embedding.tolist()
                        return embedding if isinstance(embedding, list) else [0.0] * 384
                    self.vectorizers[backend] = default_vectorize
                else:
                    logging.warning(f"⚠️ Unknown backend: {backend}")
                    continue
                    
                logging.info(f"✅ {backend} vectorizer loaded")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load {backend} vectorizer: {e}")
    
    def vectorize_parallel(self, text: str) -> Dict[str, BackendResult]:
        """
        Vectorize text using all available backends in parallel.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            Dictionary mapping backend names to their results
        """
        if not self.hybrid_mode:
            # Use primary backend only
            primary = self.config.get('vector_backend', 'default')
            return self._vectorize_single(text, primary)
        
        results = {}
        
        # Execute vectorization in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all vectorization tasks
            future_to_backend = {
                executor.submit(self._vectorize_with_timing, backend, text): backend
                for backend in self.vectorizers.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_backend, timeout=self.timeout):
                backend = future_to_backend[future]
                try:
                    result = future.result()
                    results[backend] = result
                except Exception as e:
                    results[backend] = BackendResult(
                        backend=backend,
                        vector=[],
                        success=False,
                        error=str(e)
                    )
                    logging.error(f"❌ {backend} vectorization failed: {e}")
        
        return results
    
    def _vectorize_single(self, text: str, backend: str) -> Dict[str, BackendResult]:
        """Vectorize using single backend (non-hybrid mode)."""
        result = self._vectorize_with_timing(backend, text)
        return {backend: result}
    
    def _vectorize_with_timing(self, backend: str, text: str) -> BackendResult:
        """Vectorize text with a specific backend and measure timing."""
        start_time = time.time()
        
        try:
            vectorizer = self.vectorizers[backend]
            vector = vectorizer(text)
            latency = time.time() - start_time
            
            return BackendResult(
                backend=backend,
                vector=vector,
                success=True,
                latency=latency
            )
        except Exception as e:
            latency = time.time() - start_time
            return BackendResult(
                backend=backend,
                vector=[],
                success=False,
                error=str(e),
                latency=latency
            )
    
    def find_similar_hybrid(self, query: str, facts: List[Dict[str, Any]], 
                           top_k: int = 10) -> List[HybridMemoryResult]:
        """
        Find similar facts using hybrid memory intelligence.
        
        Args:
            query: Search query
            facts: List of fact dictionaries with content and metadata
            top_k: Number of results to return
            
        Returns:
            List of hybrid memory results with source attribution
        """
        # Vectorize query using all backends
        query_vectors = self.vectorize_parallel(query)
        
        if not any(result.success for result in query_vectors.values()):
            logging.error("❌ All vectorization backends failed")
            return []
        
        # Route query based on strategy
        if self.hybrid_strategy == 'ensemble':
            return self._ensemble_search(query_vectors, facts, top_k)
        elif self.hybrid_strategy == 'priority':
            return self._priority_search(query_vectors, facts, top_k)
        elif self.hybrid_strategy == 'contextual':
            return self._contextual_search(query, query_vectors, facts, top_k)
        else:
            logging.warning(f"Unknown strategy {self.hybrid_strategy}, using ensemble")
            return self._ensemble_search(query_vectors, facts, top_k)
    
    def _ensemble_search(self, query_vectors: Dict[str, BackendResult], 
                        facts: List[Dict[str, Any]], top_k: int) -> List[HybridMemoryResult]:
        """
        Ensemble search: combine results from all backends with weighted voting.
        """
        all_results = []
        
        # Search with each successful backend
        for backend, query_result in query_vectors.items():
            if not query_result.success:
                continue
                
            backend_results = self._search_with_backend(
                query_result.vector, facts, backend
            )
            all_results.extend(backend_results)
        
        # Fuse results using weighted scoring
        fused_results = self._fuse_results(all_results, top_k)
        
        return fused_results
    
    def _priority_search(self, query_vectors: Dict[str, BackendResult],
                        facts: List[Dict[str, Any]], top_k: int) -> List[HybridMemoryResult]:
        """
        Priority search: use backends in preference order with fallback.
        """
        backend_priority = ['default', 'hrrformer', 'vecsymr']
        
        for backend in backend_priority:
            if backend in query_vectors and query_vectors[backend].success:
                results = self._search_with_backend(
                    query_vectors[backend].vector, facts, backend
                )
                return results[:top_k]
        
        return []
    
    def _contextual_search(self, query: str, query_vectors: Dict[str, BackendResult],
                          facts: List[Dict[str, Any]], top_k: int) -> List[HybridMemoryResult]:
        """
        Contextual search: route to best backend based on query characteristics.
        """
        # Analyze query to determine best backend
        best_backend = self._select_backend_for_query(query)
        
        if best_backend in query_vectors and query_vectors[best_backend].success:
            primary_results = self._search_with_backend(
                query_vectors[best_backend].vector, facts, best_backend
            )
            
            # Supplement with other backends if needed
            if len(primary_results) < top_k:
                for backend, query_result in query_vectors.items():
                    if backend != best_backend and query_result.success:
                        supplementary = self._search_with_backend(
                            query_result.vector, facts, backend
                        )
                        primary_results.extend(supplementary)
            
            return primary_results[:top_k]
        
        # Fallback to ensemble if contextual routing fails
        return self._ensemble_search(query_vectors, facts, top_k)
    
    def _select_backend_for_query(self, query: str) -> str:
        """
        Select the best backend for a query based on its characteristics.
        """
        query_lower = query.lower()
        
        # Mathematical or logical reasoning -> HRRFormer
        if any(word in query_lower for word in ['calculate', 'compute', 'logic', 'if', 'then']):
            return 'hrrformer'
        
        # Analogical or comparative queries -> VecSymR
        if any(word in query_lower for word in ['like', 'similar', 'analogy', 'compare', 'relationship']):
            return 'vecsymr'
        
        # Default to semantic search for general queries
        return 'default'
    
    def _search_with_backend(self, query_vector: List[float], 
                           facts: List[Dict[str, Any]], backend: str) -> List[HybridMemoryResult]:
        """
        Search facts using a specific backend vector.
        """
        results = []
        query_vec = np.array(query_vector)
        
        for fact in facts:
            # Get fact vector (assume it's stored or compute it)
            fact_vector = self._get_or_compute_fact_vector(fact, backend)
            if fact_vector is None:
                continue
            
            fact_vec = np.array(fact_vector)
            
            # Compute similarity
            try:
                if len(query_vec) == len(fact_vec):
                    similarity = 1 - cosine(query_vec, fact_vec)
                else:
                    # Handle dimension mismatch by padding/truncating
                    min_dim = min(len(query_vec), len(fact_vec))
                    similarity = 1 - cosine(query_vec[:min_dim], fact_vec[:min_dim])
            except:
                similarity = 0.0
            
            if similarity > self.confidence_threshold:
                result = HybridMemoryResult(
                    content=fact.get('content', ''),
                    vector=fact_vector,
                    confidence=similarity,
                    recency_score=self._compute_recency_score(fact),
                    semantic_overlap=similarity,  # Will be refined in fusion
                    source_backend=backend,
                    original_score=similarity,
                    hybrid_score=0.0,  # Will be computed in fusion
                    metadata=fact
                )
                results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def _get_or_compute_fact_vector(self, fact: Dict[str, Any], backend: str) -> Optional[List[float]]:
        """
        Get or compute the vector for a fact using the specified backend.
        """
        # Check if vector is cached for this backend
        vector_key = f'vector_{backend}'
        if vector_key in fact:
            return fact[vector_key]
        
        # Compute vector
        content = fact.get('content', '')
        if not content:
            return None
        
        try:
            vectorizer = self.vectorizers[backend]
            vector = vectorizer(content)
            # Cache for future use
            fact[vector_key] = vector
            return vector
        except Exception as e:
            logging.warning(f"Failed to vectorize fact with {backend}: {e}")
            return None
    
    def _compute_recency_score(self, fact: Dict[str, Any]) -> float:
        """
        Compute recency score for a fact (0-1, where 1 is most recent).
        """
        try:
            timestamp = fact.get('timestamp', time.time())
            if isinstance(timestamp, str):
                from datetime import datetime
                timestamp = datetime.fromisoformat(timestamp).timestamp()
            
            # Recency score based on age (24 hours = 1.0, older = lower)
            age_hours = (time.time() - timestamp) / 3600
            recency = max(0.0, 1.0 - age_hours / 24.0)
            return recency
        except:
            return 0.5  # Default neutral recency
    
    def _fuse_results(self, all_results: List[HybridMemoryResult], 
                     top_k: int) -> List[HybridMemoryResult]:
        """
        Fuse results from multiple backends using intelligent weighting.
        """
        # Group results by content to avoid duplicates
        content_groups = {}
        for result in all_results:
            content = result.content
            if content not in content_groups:
                content_groups[content] = []
            content_groups[content].append(result)
        
        # Compute hybrid scores for each unique content
        fused_results = []
        for content, backend_results in content_groups.items():
            fused_result = self._fuse_content_results(backend_results)
            if fused_result:
                fused_results.append(fused_result)
        
        # Sort by hybrid score and return top_k
        fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return fused_results[:top_k]
    
    def _fuse_content_results(self, backend_results: List[HybridMemoryResult]) -> Optional[HybridMemoryResult]:
        """
        Fuse multiple backend results for the same content.
        """
        if not backend_results:
            return None
        
        # Use the result with highest confidence as base
        base_result = max(backend_results, key=lambda x: x.confidence)
        
        # Compute weighted hybrid score
        total_weight = 0
        weighted_score = 0
        
        for result in backend_results:
            backend_weight = self.weights.get(result.source_backend, 0.1)
            
            # Component scores
            confidence_score = result.confidence
            recency_score = result.recency_score
            source_score = backend_weight
            
            # Semantic overlap (computed across backends)
            semantic_score = self._compute_semantic_overlap(backend_results)
            
            # Combine scores
            combined_score = (
                confidence_score * self.semantic_weight +
                recency_score * self.recency_weight +
                source_score * self.source_weight +
                semantic_score * (1.0 - self.semantic_weight - self.recency_weight - self.source_weight)
            )
            
            weighted_score += combined_score * backend_weight
            total_weight += backend_weight
        
        if total_weight > 0:
            hybrid_score = weighted_score / total_weight
        else:
            hybrid_score = base_result.confidence
        
        # Create fused result
        fused_result = HybridMemoryResult(
            content=base_result.content,
            vector=base_result.vector,
            confidence=max(r.confidence for r in backend_results),
            recency_score=base_result.recency_score,
            semantic_overlap=self._compute_semantic_overlap(backend_results),
            source_backend=f"hybrid_{len(backend_results)}",
            original_score=base_result.original_score,
            hybrid_score=hybrid_score,
            metadata={
                **base_result.metadata,
                'fusion_info': {
                    'backends': [r.source_backend for r in backend_results],
                    'scores': [r.confidence for r in backend_results],
                    'fusion_strategy': self.hybrid_strategy
                }
            }
        )
        
        return fused_result
    
    def _compute_semantic_overlap(self, backend_results: List[HybridMemoryResult]) -> float:
        """
        Compute semantic overlap score between backend results.
        """
        if len(backend_results) < 2:
            return 1.0
        
        # Compute pairwise similarities between backend vectors
        similarities = []
        for i in range(len(backend_results)):
            for j in range(i + 1, len(backend_results)):
                vec1 = np.array(backend_results[i].vector)
                vec2 = np.array(backend_results[j].vector)
                
                try:
                    if len(vec1) == len(vec2):
                        sim = 1 - cosine(vec1, vec2)
                    else:
                        min_dim = min(len(vec1), len(vec2))
                        sim = 1 - cosine(vec1[:min_dim], vec2[:min_dim])
                    similarities.append(sim)
                except:
                    similarities.append(0.5)
        
        return np.mean(similarities) if similarities else 0.5
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hybrid memory performance.
        """
        return {
            "config": self.config,
            "active_backends": list(self.vectorizers.keys()),
            "backend_weights": self.weights,
            "hybrid_mode": self.hybrid_mode,
            "strategy": self.hybrid_strategy,
            "fusion_parameters": {
                "confidence_threshold": self.confidence_threshold,
                "recency_weight": self.recency_weight,
                "semantic_weight": self.semantic_weight,
                "source_weight": self.source_weight
            }
        }