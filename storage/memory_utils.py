#!/usr/bin/env python3
"""
Smart memory utilities with centralized configuration
"""

import logging
import os
import re
import yaml
import requests
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from difflib import SequenceMatcher

from config.settings import (
    CATEGORY_ORDER,
    CATEGORY_PATTERNS,
    CONFIDENCE_ICONS,
    CONFIDENCE_THRESHOLDS,
    CONTRADICTION_SCORE_THRESHOLD,
    DEFAULT_ENTITY_CATEGORIES,
    DEFAULT_PERSONALITY,
    DEFAULT_VALUES,
    MEMORY_CONFIG,
    PERSONALITY_ADAPTATION,
    PERSONALITY_PROFILES,
    # REFERENCE_NEGATIVES,
    # REFERENCE_POSITIVES,
    # SYNONYM_MAP,
    VOLATILITY_ICONS,
    VOLATILITY_THRESHOLDS,
    QUERY_PATTERNS,  # e.g., [r'what.*(color|make|from|when)', ...]
)
from scripts.embedder import embed
from storage.formatters import format_memory_section

# Embedding cache for performance
_embedding_cache = {}
_cache_hits = 0
_cache_misses = 0

# LLM semantic analysis cache for performance
_llm_cache = {}
_llm_cache_hits = 0
_llm_cache_misses = 0

# Load config.yaml for LLM-driven settings
def _load_memory_config():
    """Load memory configuration from config.yaml - NO FALLBACK HARDCODING"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        memory_config = config.get('memory', {})
        if not memory_config:
            raise RuntimeError("Memory configuration section missing from config.yaml")
        return memory_config
    except Exception as e:
        logging.error(f"Failed to load memory config from config.yaml: {e}")
        raise RuntimeError(f"Memory configuration is required but could not be loaded: {e}")

_memory_config = _load_memory_config()

def _call_ollama_llm(prompt: str, cache_key: str = None) -> str:
    """
    Call Ollama's Mistral model for semantic analysis.
    
    Args:
        prompt: The prompt to send to the LLM
        cache_key: Optional cache key for performance
    
    Returns:
        LLM response text
    """
    global _llm_cache, _llm_cache_hits, _llm_cache_misses
    
    # Check cache first
    if cache_key and _memory_config.get('performance', {}).get('enable_llm_caching', True):
        if cache_key in _llm_cache:
            _llm_cache_hits += 1
            return _llm_cache[cache_key]
    
    try:
        # Get Ollama host from config.yaml - NO FALLBACK
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ollama_host = config.get('network', {}).get('ollama_host')
        if not ollama_host:
            raise RuntimeError("Ollama host configuration missing from config.yaml")
        # Quick health check to avoid long timeouts when unavailable
        try:
            import requests as _rq
            r = _rq.get(f"{ollama_host}/api/tags", timeout=2)
            if r.status_code != 200:
                return ""
        except Exception:
            return ""

        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent semantic judgments
                    "top_p": 0.9
                }
            },
            timeout=(3.5, 8)
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            _llm_cache_misses += 1
            
            # Cache the result
            if cache_key and _memory_config.get('performance', {}).get('enable_llm_caching', True):
                _llm_cache[cache_key] = result
                
                # Simple cache size management
                if len(_llm_cache) > 1000:
                    # Remove oldest 20% of entries
                    keys_to_remove = list(_llm_cache.keys())[:200]
                    for key in keys_to_remove:
                        del _llm_cache[key]
            
            return result
        else:
            logging.warning(f"Ollama API error {response.status_code}: {response.text}")
            return ""
            
    except Exception as e:
        logging.warning(f"Failed to call Ollama LLM: {e}")
        return ""


def get_cached_embedding(text: str) -> np.ndarray:
    """
    Get embedding with caching for performance.

    Args:
        text: Text to embed

    Returns:
        Cached or computed embedding
    """
    global _cache_hits, _cache_misses

    if not text:
        return np.zeros(384, dtype="float32")

    text_key = text.lower().strip()

    if text_key in _embedding_cache:
        _cache_hits += 1
        return _embedding_cache[text_key]

    _cache_misses += 1
    embedding = embed(text)
    _embedding_cache[text_key] = embedding

    # Limit cache size to prevent memory issues
    if len(_embedding_cache) > 10000:
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(_embedding_cache.keys())[:1000]
        for key in oldest_keys:
            del _embedding_cache[key]

    return embedding


def get_embedding_cache_stats() -> Dict[str, int]:
    """Get embedding cache statistics"""
    return {
        "cache_size": len(_embedding_cache),
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate": (
            _cache_hits / (_cache_hits + _cache_misses)
            if (_cache_hits + _cache_misses) > 0
            else 0.0
        ),
    }


def clear_embedding_cache():
    """Clear the embedding cache"""
    global _embedding_cache, _cache_hits, _cache_misses
    _embedding_cache.clear()
    _cache_hits = 0
    _cache_misses = 0

def get_llm_cache_stats() -> Dict[str, int]:
    """Get LLM semantic analysis cache statistics"""
    return {
        "cache_size": len(_llm_cache),
        "cache_hits": _llm_cache_hits,
        "cache_misses": _llm_cache_misses,
        "hit_rate": (
            _llm_cache_hits / (_llm_cache_hits + _llm_cache_misses)
            if (_llm_cache_hits + _llm_cache_misses) > 0
            else 0.0
        ),
    }

def clear_llm_cache():
    """Clear the LLM semantic analysis cache"""
    global _llm_cache, _llm_cache_hits, _llm_cache_misses
    _llm_cache.clear()
    _llm_cache_hits = 0
    _llm_cache_misses = 0

def get_combined_cache_stats() -> Dict[str, dict]:
    """Get combined statistics for both embedding and LLM caches"""
    return {
        "embedding_cache": get_embedding_cache_stats(),
        "llm_cache": get_llm_cache_stats()
    }


@dataclass
class TripletFact:
    """Represents a subject-predicate-object triplet fact with metadata"""

    id: int
    subject: str
    predicate: str
    object: str  # Keep as 'object' - the field accessor will use getattr
    source_message_id: Optional[int] = None  # ID of source message
    timestamp: str = ""
    frequency: int = 1
    contradiction_score: float = 0.0
    volatility_score: float = 0.0
    confidence: float = 1.0
    user_profile_id: Optional[str] = None  # User profile ID for fact scoping
    session_id: Optional[str] = None  # Session ID for fact scoping
    
    @property
    def object_(self):
        """Compatibility property for existing code that uses object_"""
        return self.object


@dataclass
class MemoryFact:
    """Enhanced memory fact with categorization"""

    value: str
    category: str
    confidence: float
    source_message_id: Optional[int] = None
    timestamp: Optional[str] = None
    volatility: float = DEFAULT_VALUES[
        "volatility_default"
    ]  # Number of changes per day since creation


def normalize_subject_references(text: str) -> str:
    """
    Normalize first-person references (I, me, my, etc.) to "you" for consistent memory storage.

    Args:
        text: Input text with potential first-person references

    Returns:
        Text with normalized subject references
    """
    # Common first-person patterns to normalize
    first_person_patterns = [
        (r"\bI\'m\b", "you are"),  # Handle contractions first
        (r"\bI\'ve\b", "you have"),
        (r"\bI\'ll\b", "you will"),
        (r"\bI\'d\b", "you would"),
        (r"\bI\'re\b", "you are"),  # Handle potential typos
        (r"\bI\b", "you"),  # Handle standalone "I" after contractions
        (r"\bme\b", "you"),
        (r"\bmy\b", "your"),
        (r"\bmine\b", "yours"),
        (r"\bmyself\b", "yourself"),
    ]

    normalized_text = text
    for pattern, replacement in first_person_patterns:
        normalized_text = re.sub(
            pattern, replacement, normalized_text, flags=re.IGNORECASE
        )

    return normalized_text


def _normalize_subject(subject: str) -> str:
    """
    Normalize subject references to canonical form for better matching.
    Handles common patterns like "love pizza", "hate [pizza]", "your cat", etc.

    Args:
        subject: Original subject string

    Returns:
        Canonical subject form
    """
    if not subject:
        return ""

    subject = subject.lower()

    # Remove brackets and their contents
    subject = re.sub(r"\[.*?\]", "", subject)
    subject = subject.strip()

    # Remove common possessive and article prefixes
    for prefix in [
        "your ",
        "my ",
        "his ",
        "her ",
        "their ",
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
    ]:
        if subject.startswith(prefix):
            subject = subject[len(prefix) :]

    # Remove trailing qualifiers
    for suffix in ["now", "today", "always", "never"]:
        if subject.endswith(suffix):
            subject = subject[: -len(suffix)]

    # Clean up any remaining whitespace
    subject = subject.strip()

    # If we have a single word, return it
    if len(subject.split()) == 1:
        return subject

    # For multi-word subjects, try to extract the main noun
    words = subject.split()

    # Common verb patterns - extract the object
    if len(words) >= 2:
        # Simple heuristic: if first word looks like a verb, take the last word as subject
        if words[0] in [
            "love",
            "hate",
            "like",
            "dislike",
            "adore",
            "despise",
            "enjoy",
            "prefer",
            "want",
            "need",
            "ate",
            "eat",
            "eats",
            "eating",
            "loved",
            "hated",
            "liked",
            "disliked",
            "adored",
            "despised",
            "enjoyed",
            "preferred",
        ]:
            # Skip articles and possessive pronouns
            for i in range(1, len(words)):
                if words[i] not in [
                    "a",
                    "an",
                    "the",
                    "your",
                    "my",
                    "his",
                    "her",
                    "their",
                ]:
                    return words[i]

    # If no clear pattern, return the last word (often the main noun)
    return words[-1] if words else ""


def _smart_normalize_subject(subject: str, fact_context: List[tuple] = None) -> str:
    """
    Smart subject normalization that learns from existing fact patterns.
    Instead of hardcoding verbs, it analyzes the actual structure of facts in the database.

    Args:
        subject: Original subject string
        fact_context: List of (subject, predicate, object) tuples from the database

    Returns:
        Canonical subject form
    """
    if not subject:
        return ""

    subject = subject.lower()

    # Remove brackets and their contents
    subject = re.sub(r"\[.*?\]", "", subject)
    subject = subject.strip()

    # Remove common possessive and article prefixes
    for prefix in [
        "your ",
        "my ",
        "his ",
        "her ",
        "their ",
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
    ]:
        if subject.startswith(prefix):
            subject = subject[len(prefix) :]

    # Remove trailing qualifiers
    for suffix in ["now", "today", "always", "never"]:
        if subject.endswith(suffix):
            subject = subject[: -len(suffix)]

    # Clean up any remaining whitespace
    subject = subject.strip()

    # If we have a single word, return it
    if len(subject.split()) == 1:
        return subject

    # For multi-word subjects, use pattern learning if context is available
    if fact_context:
        return _learn_subject_pattern(subject, fact_context)

    # Fallback to simple heuristic
    words = subject.split()
    if len(words) >= 2:
        # Take the last word as the subject (often the main noun)
        return words[-1]

    return subject


def _learn_subject_pattern(subject: str, fact_context: List[tuple]) -> str:
    """
    Learn subject patterns from existing facts in the database.
    """
    words = subject.split()

    # Look for similar patterns in existing facts
    for existing_subject, existing_predicate, existing_object in fact_context:
        existing_words = existing_subject.lower().split()

        # If we have similar word patterns, use the same extraction logic
        if len(existing_words) >= 2 and len(words) >= 2:
            # Check if first word is the same (likely a verb)
            if existing_words[0] == words[0]:
                # Use the same extraction pattern
                if len(existing_words) >= 2:
                    return existing_words[1]  # Take second word as subject
                elif len(words) >= 2:
                    return words[1]

    # Default: take the last word
    return words[-1] if words else subject


def _normalize_subject_with_embeddings(
    subject: str, existing_subjects: List[str]
) -> str:
    """
    Normalize subject using embedding similarity to find the most similar existing subject.
    This is the preferred method when you have existing subjects to compare against.

    Args:
        subject: Original subject string
        existing_subjects: List of existing subjects to compare against

    Returns:
        Most similar existing subject, or original subject if no good match
    """
    if not existing_subjects:
        return _normalize_subject(subject)

    try:
        import numpy as np

        from scripts.embedder import embed, most_similar

        # Get embedding for the subject
        subject_embedding = embed(subject)
        if np.all(subject_embedding == 0):
            logging.warning(
                f"Warning: Embedding failed for subject '{subject}' in _normalize_subject_with_embeddings."
            )

        # Find most similar existing subject
        best_match = None
        best_similarity = 0.0

        for existing_subject in existing_subjects:
            existing_embedding = embed(existing_subject)
            if np.all(existing_embedding == 0):
                logging.warning(
                    f"Warning: Embedding failed for existing subject '{existing_subject}' in _normalize_subject_with_embeddings."
                )
            # most_similar returns list of (token, score) tuples
            similarities = most_similar(existing_embedding, topk=1)
            if similarities:
                similarity = similarities[0][1]  # Get the score from the first result
                if (
                    similarity > best_similarity
                    and similarity > CONFIDENCE_THRESHOLDS["high"]
                ):  # High similarity threshold
                    best_similarity = similarity
                    best_match = existing_subject

        return best_match if best_match else _normalize_subject(subject)

    except Exception as e:
        logging.warning(f"Embedding-based normalization failed: {e}")
        return _normalize_subject(subject)


def _are_antonyms(word1: str, word2: str) -> bool:
    """Check if two words are antonyms using LLM-driven semantic analysis"""
    if not word1 or not word2:
        return False
    
    # Normalize words
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    
    # Quick check for identical words
    if word1 == word2:
        return False
    
    # Hardcoded antonym pairs for common cases
    antonym_pairs = [
        ("like", "hate"), ("likes", "hates"),
        ("love", "hate"), ("loves", "hates"),
        ("like", "dislike"), ("likes", "dislikes"),
        ("love", "dislike"), ("loves", "dislikes"),
        ("enjoy", "hate"), ("enjoys", "hates"),
        ("prefer", "avoid"), ("prefers", "avoids"),
        ("good", "bad"), ("hot", "cold"),
        ("yes", "no"), ("true", "false")
    ]
    
    # Check both directions
    for w1, w2 in antonym_pairs:
        if (word1 == w1 and word2 == w2) or (word1 == w2 and word2 == w1):
            return True
    
    # Create cache key
    cache_key = f"antonym:{word1}:{word2}"
    
    # Get prompt template from config
    prompt_template = _memory_config.get('normalization', {}).get('llm_prompts', {}).get(
        'antonym_check', "Are {word1} and {word2} antonyms (opposite meaning)? Answer ONLY: yes or no"
    )
    
    prompt = prompt_template.format(word1=word1, word2=word2)
    
    # Call LLM
    response = _call_ollama_llm(prompt, cache_key)
    
    # Parse response - handle variations and quotes
    response_clean = response.lower().strip().strip('"\'').strip('.')
    return 'yes' in response_clean and 'no' not in response_clean


def detect_contradictions(
    facts: List[TripletFact],
) -> List[Tuple[TripletFact, TripletFact, float]]:
    """
    Detect contradictions between facts in the given list.

    Args:
        facts: List of TripletFact objects to check for contradictions

    Returns:
        List of (fact1, fact2, contradiction_score) tuples
    """
    contradictions = []

    for i, fact1 in enumerate(facts):
        for j, fact2 in enumerate(facts[i + 1 :], i + 1):
            # Skip if facts are about different subjects
            if fact1.subject.lower() != fact2.subject.lower():
                continue

            # Calculate contradiction score
            score = calculate_contradiction_score(fact1, fact2)

            # Only include significant contradictions
            if score > CONTRADICTION_SCORE_THRESHOLD:
                contradictions.append((fact1, fact2, score))

    return contradictions


def calculate_contradiction_score(fact1, fact2) -> float:
    """Calculate contradiction score between two facts using embedding similarity"""
    # 🩹 Handle case where tuples are passed instead of TripletFact objects
    if isinstance(fact1, tuple):
        from storage.memory_log import TripletFact

        fact1 = TripletFact(
            id=0,
            subject=fact1[0],
            predicate=fact1[1],
            object=fact1[2],
            frequency=1,
            timestamp="",
        )
    if isinstance(fact2, tuple):
        from storage.memory_log import TripletFact

        fact2 = TripletFact(
            id=0,
            subject=fact2[0],
            predicate=fact2[1],
            object=fact2[2],
            frequency=1,
            timestamp="",
        )

    try:
        # Check for None or empty objects
        if not fact1.object or not fact2.object:
            return 0.0

        # Normalize subjects for comparison
        subj1 = _normalize_subject(fact1.subject)
        subj2 = _normalize_subject(fact2.subject)

        # If subjects don't match, no contradiction
        if subj1 != subj2:
            return 0.0

        # Check predicates and objects for contradictions
        pred1 = fact1.predicate.lower().strip()
        pred2 = fact2.predicate.lower().strip()
        obj1 = fact1.object.lower().strip()
        obj2 = fact2.object.lower().strip()

        # Case 1: Same predicate, different objects (potential contradiction)
        if pred1 == pred2:
            # Check for direct antonyms in objects
            if _are_antonyms(obj1, obj2):
                return CONFIDENCE_THRESHOLDS["very_high"]
            multi_value_predicates = ["like", "likes", "enjoy", "enjoys", "prefer", "prefers", "have", "has"]
            if pred1 in multi_value_predicates:
                return 0.0
            # Use cached embedding similarity for objects
            obj_sim = cosine_similarity(
                get_cached_embedding(obj1), get_cached_embedding(obj2)
            )
            # Low similarity in objects with same subject/predicate = contradiction
            if obj_sim < CONFIDENCE_THRESHOLDS["low"]:
                return 1.0 - obj_sim  # Higher contradiction for more different objects
            return 0.0

        # Case 2: Different predicates, same object (potential contradiction)
        elif obj1 == obj2:
            # Check for antonym predicates
            if _are_antonyms(pred1, pred2):
                return CONFIDENCE_THRESHOLDS["very_high"]

            # Use cached embedding similarity for predicates
            pred_sim = cosine_similarity(
                get_cached_embedding(pred1), get_cached_embedding(pred2)
            )

            # Low similarity in predicates with same subject/object = contradiction
            if pred_sim < CONFIDENCE_THRESHOLDS["low"]:
                return 1.0 - pred_sim

        # Case 3: Different predicates and objects - check for semantic contradictions
        else:
            # Check if predicates are antonyms
            if _are_antonyms(pred1, pred2):
                # If predicates are antonyms, check if objects are similar
                obj_sim = cosine_similarity(
                    get_cached_embedding(obj1), get_cached_embedding(obj2)
                )
                if (
                    obj_sim > CONFIDENCE_THRESHOLDS["medium"]
                ):  # High object similarity with antonym predicates
                    return CONFIDENCE_THRESHOLDS["high"]

            # New: Check for positive vs negative sentiments
            positive_preds = ["like", "likes", "love", "loves", "enjoy", "enjoys", "prefer", "prefers"]
            negative_preds = ["hate", "hates", "dislike", "dislikes", "avoid", "avoids"]
            if (pred1 in positive_preds and pred2 in negative_preds) or (pred1 in negative_preds and pred2 in positive_preds):
                obj_sim = cosine_similarity(
                    get_cached_embedding(obj1), get_cached_embedding(obj2)
                )
                if obj_sim > CONFIDENCE_THRESHOLDS["medium"]:
                    return CONFIDENCE_THRESHOLDS["high"]

            # Check if objects are antonyms
            if _are_antonyms(obj1, obj2):
                # If objects are antonyms, check if predicates are similar
                pred_sim = cosine_similarity(
                    get_cached_embedding(pred1), get_cached_embedding(pred2)
                )
                if (
                    pred_sim > CONFIDENCE_THRESHOLDS["medium"]
                ):  # High predicate similarity with antonym objects
                    return CONFIDENCE_THRESHOLDS["high"]

        return 0.0

    except Exception as e:
        # Fallback to simple word overlap if embedding fails
        logging.warning(f"Embedding similarity failed, using fallback: {e}")
        return calculate_simple_contradiction_score(fact1, fact2)


def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between -1.0 and 1.0
    """
    import numpy as np

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_simple_contradiction_score(fact1, fact2) -> float:
    """
    Fallback contradiction detection using simple word overlap.

    Args:
        fact1: First TripletFact or tuple
        fact2: Second TripletFact or tuple

    Returns:
        Contradiction score between 0.0 and 1.0
    """
    # 🩹 Handle case where tuples are passed instead of TripletFact objects
    if isinstance(fact1, tuple):
        from storage.memory_log import TripletFact

        fact1 = TripletFact(
            id=0,
            subject=fact1[0],
            predicate=fact1[1],
            object=fact1[2],
            frequency=1,
            timestamp="",
        )
    if isinstance(fact2, tuple):
        from storage.memory_log import TripletFact

        fact2 = TripletFact(
            id=0,
            subject=fact2[0],
            predicate=fact2[1],
            object=fact2[2],
            frequency=1,
            timestamp="",
        )

    subj1 = _normalize_subject(fact1.subject)
    subj2 = _normalize_subject(fact2.subject)
    pred1 = fact1.predicate.lower().strip()
    pred2 = fact2.predicate.lower().strip()
    obj1 = fact1.object.lower().strip()
    obj2 = fact2.object.lower().strip()

    # Check for exact matches in subject and predicate
    if subj1 == subj2 and pred1 == pred2 and obj1 != obj2:

        # Calculate object similarity
        obj1_words = set(obj1.split())
        obj2_words = set(obj2.split())

        if obj1_words and obj2_words:
            overlap = len(obj1_words & obj2_words)
            total = len(obj1_words | obj2_words)
            similarity = overlap / total if total > 0 else 0.0

            # Low similarity = high contradiction
            return 1.0 - similarity

    return 0.0


def calculate_volatility_score(fact_history: List[Dict]) -> float:
    """
    Calculate volatility score based on fact change history.

    Args:
        fact_history: List of fact change records

    Returns:
        Volatility score between 0.0 and 1.0
    """
    if len(fact_history) < 1:
        return 0.0

    # Use frequency as a proxy for volatility
    # Higher frequency suggests more changes/updates
    total_frequency = sum(record.get("frequency", 1) for record in fact_history)

    # Simple volatility calculation based on frequency
    # More frequent facts are considered more volatile
    volatility = min(1.0, (total_frequency - 1) / 3.0)  # Normalize to 0-1 scale

    return volatility


def resolve_conflicts(
    new_fact: Tuple[str, str, str],
    existing_facts: List[TripletFact],
    strategy: str = "most_recent",
) -> Tuple[bool, Optional[TripletFact]]:
    """
    Resolve conflicts between new and existing facts.

    Args:
        new_fact: New fact tuple (subject, predicate, object)
        existing_facts: List of existing facts
        strategy: Conflict resolution strategy ("most_recent", "highest_frequency", "manual")

    Returns:
        Tuple of (should_store_new, conflicting_fact_if_any)
    """
    contradictions = detect_contradictions(existing_facts)

    if not contradictions:
        return True, None

    # Find the most contradictory fact
    most_contradictory = max(contradictions, key=lambda x: x[2])
    conflicting_fact, contradiction_score = most_contradictory

    if strategy == "most_recent":
        # Always store new fact, mark old one as contradicted
        return True, conflicting_fact
    elif strategy == "highest_frequency":
        # Keep the fact with higher frequency
        if conflicting_fact.frequency > 1:  # New fact starts with frequency 1
            return False, conflicting_fact
        else:
            return True, conflicting_fact
    else:  # manual
        # Return both for manual review
        return True, conflicting_fact


def categorize_fact(text: str, entity: str) -> str:
    """
    Categorize a fact based on its content and entity type using centralized mappings

    Args:
        text: The original text
        entity: The extracted entity type

    Returns:
        Category string
    """
    text_lower = text.lower()

    # Check direct entity mapping first
    if entity in DEFAULT_ENTITY_CATEGORIES:
        return DEFAULT_ENTITY_CATEGORIES[entity]

    # Check pattern-based categorization
    for category, keywords in CATEGORY_PATTERNS.items():
        for keyword in keywords:
            if keyword in text_lower or keyword in entity.lower():
                return category

    # Default to misc
    return CATEGORY_ORDER[-1]  # Use the last category as the default fallback


def build_memory_context(triplets: List[TripletFact], max_tokens: int = 512) -> str:
    """
    Build a smart memory context from triplet facts, grouped by predicate.
    Args:
        triplets: List of TripletFact
        max_tokens: Maximum tokens to include in context
    Returns:
        Formatted memory context string
    """
    if not triplets:
        return "No memory facts available."
    grouped = defaultdict(list)
    for fact in triplets:
        grouped[fact.predicate].append(fact)
    # Sort by frequency (descending) and recency
    sorted_predicates = sorted(grouped.keys())
    context_lines = []
    tokens_used = 0
    for pred in sorted_predicates:
        # Sort by decayed confidence (highest first) and recency
        facts = sorted(
            grouped[pred],
            key=lambda f: (-getattr(f, "decayed_confidence", 1.0), f.timestamp),
        )
        for fact in facts:
            decayed_conf = getattr(fact, "decayed_confidence", 1.0)
            line = f"{fact.subject} {fact.predicate} {fact.object} (confidence: {decayed_conf:.2f}, seen {fact.frequency}×, last: {fact.timestamp.split()[0]})"
            token_estimate = len(line.split()) + 2
            if tokens_used + token_estimate > max_tokens:
                break
            context_lines.append(line)
            tokens_used += token_estimate
    if not context_lines:
        return "No memory facts available."
    return "\n".join(context_lines)


def build_prompt(
    user_message: str,
    memory_log,
    max_tokens: int = 512,
    memory_mode: str = "MAC",
    personality: str = "neutral",
) -> str:
    """
    Build a complete prompt with smart memory injection using triplet facts.
    Supports different memory routing modes: MAC, MAG, MEL and personality traits.

    Args:
        user_message: The user's message
        memory_log: MemoryLog instance to get triplets from
        max_tokens: Maximum tokens for memory context
        memory_mode: Memory routing mode ("MAC", "MAG", or "MEL")
        personality: Personality type for memory biasing

    Returns:
        Complete system prompt
    """
    from config.settings import DEFAULT_MEMORY_MODE, MEMORY_ROUTING_MODES

    # Validate memory mode
    if memory_mode not in MEMORY_ROUTING_MODES:
        memory_mode = DEFAULT_MEMORY_MODE
        logging.warning(
            f"{CONFIDENCE_ICONS['medium']} Invalid memory mode, using default: {DEFAULT_MEMORY_MODE}"
        )

    # Validate personality
    if personality not in PERSONALITY_PROFILES:
        personality = DEFAULT_PERSONALITY
        logging.warning(
            f"{CONFIDENCE_ICONS['medium']} Invalid personality, using default: {DEFAULT_PERSONALITY}"
        )

    personality_config = PERSONALITY_PROFILES[personality]

    # Use personality-aware decay to get facts
    triplets = memory_log.get_facts_with_personality_decay(
        personality=personality, volatility_weight=CONFIDENCE_THRESHOLDS["low"]
    )

    # Base system prompt with enhanced validation
    base_prompt = (
        "You are MeRNSTA, a blunt memory AI. Use short, clear answers. "
        "No fluff, no fake politeness, no disclaimers like 'as an AI...' — ever. "
        "Speak how the user speaks: direct, casual, occasionally sarcastic if fitting. "
        "If facts contradict, just point it out and ask which is right. "
        "Avoid overexplaining. Never repeat facts unless asked."
    )
    system_prompt = enhance_system_prompt_with_validation(base_prompt)

    # Add personality-specific prompt addition
    if personality_config["system_prompt_addition"]:
        system_prompt += f" {personality_config['system_prompt_addition']}"

    if triplets:
        # Use smart memory context that filters by relevance
        memory_context = build_smart_memory_context(triplets, user_message, max_tokens, memory_log=memory_log)
        memory_section = format_memory_section(memory_context.split("\n"))

        # Detect contradictions in relevant facts
        contradictions = detect_contradictions(triplets)
        contradiction_summary = ""

        if contradictions:
            # Filter contradictions to only those relevant to the current query
            relevant_contradictions = []
            for fact1, fact2, score in contradictions:
                # Check if either fact is relevant to the current query
                if any(
                    word in fact1.subject.lower() or word in fact1.object.lower()
                    for word in user_message.lower().split()
                    if len(word) > 2
                ) or any(
                    word in fact2.subject.lower() or word in fact2.object.lower()
                    for word in user_message.lower().split()
                    if len(word) > 2
                ):
                    relevant_contradictions.append((fact1, fact2, score))

            if relevant_contradictions:
                # Sort by contradiction score (highest first)
                relevant_contradictions.sort(key=lambda x: x[2], reverse=True)
                # Take the most significant contradiction
                fact1, fact2, score = relevant_contradictions[0]
                contradiction_summary = format_contradiction_summary(
                    fact1, fact2, score
                )

        # Check for volatile facts
        warnings = []
        high_volatility_facts = [
            f
            for f in triplets
            if f.volatility_score > personality_config["volatility_threshold"]
        ]

        if high_volatility_facts:
            warnings.append(
                f"{VOLATILITY_ICONS['high']} Note: {len(high_volatility_facts)} facts are highly volatile (frequently changing). Consider asking for confirmation."
            )

        warning_text = "\n".join(warnings) if warnings else ""

        # Apply mode-specific prompt construction
        if memory_mode == "MAC":
            # Memory as Context: Use facts as reference only
            system_prompt += f"""

{memory_section}

{contradiction_summary}

{warning_text}

Respond to the message below using relevant memory as context. Do not repeat facts unnecessarily. Always assume facts are true unless context changes. If you notice contradictions, acknowledge them and ask for clarification."""

        elif memory_mode == "MAG":
            # Memory as Generation: Inject facts inline into attention
            system_prompt += f"""

{memory_section}

{contradiction_summary}

{warning_text}

Consider these facts directly when generating your response. Let them guide your thinking process. If facts contradict, acknowledge and ask for clarification."""

        elif memory_mode == "MEL":
            # Memory as Everything: Aggressive memory-only generation
            system_prompt += f"""

{memory_section}

{contradiction_summary}

{warning_text}

Base your response entirely on the memory facts above. Prioritize memory over general knowledge. If memory is insufficient, say so clearly."""

        # Add mode-specific personality adjustments
        if memory_mode == "MEL":
            system_prompt += " Be more assertive about memory-based responses."
        elif memory_mode == "MAG":
            system_prompt += " Balance memory with general knowledge appropriately."

    else:
        system_prompt += "\n\nNo previous memory context available. Respond based on your general knowledge."

    system_prompt += f"\n\nUser: {user_message}"
    return system_prompt.strip()


def format_facts_for_display(triplets: List[TripletFact]) -> str:
    """
    Format triplet facts for display in console/UI.
    Args:
        triplets: List of TripletFact
    Returns:
        Formatted string for display
    """
    if not triplets:
        return "No facts available."
    grouped = defaultdict(list)
    for fact in triplets:
        grouped[fact.predicate].append(fact)
    lines = []
    for pred in sorted(grouped.keys()):
        lines.append(f"\n{pred.title()}:")
        # Use weighted recency + confidence ranking for facts within each predicate group
        facts_in_group = rank_facts(grouped[pred])
        for fact in facts_in_group:
            line = f"  {fact.subject} {fact.predicate} {fact.object} (seen {fact.frequency}×, last: {fact.timestamp.split()[0]})"
            lines.append(line)
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using Ollama tokenizer."""
    try:
        from utils.ollama_tokenizer import count_tokens
        return count_tokens(text)
    except ImportError:
        # Fallback to character-based estimation
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


def filter_facts_by_relevance(
    facts: List[TripletFact], query: str, max_facts: int = None
) -> List[TripletFact]:
    """
    Filter facts by relevance to a query using semantic similarity and subject matching.
    Domain-agnostic implementation that works for any subject matter.

    Args:
        facts: List of all facts
        query: Query to match against
        max_facts: Maximum number of facts to return (uses config default if None)

    Returns:
        Filtered list of relevant facts
    """
    if not facts:
        return []

    # Use config default if max_facts not specified
    if max_facts is None:
        max_facts = MEMORY_CONFIG["max_facts_per_prompt"]

    query_lower = query.lower()
    query_words = set(query_lower.split())

    # Score facts by relevance using domain-agnostic criteria
    scored_facts = []
    for fact in facts:
        score = 0

        # Exact phrase matching gets highest priority (domain-agnostic)
        if query_lower in fact.subject.lower():
            score += 25
        elif query_lower in fact.object.lower():
            score += 20
        elif query_lower in fact.predicate.lower():
            score += 15

        # Word-level matching (domain-agnostic)
        subject_matches = sum(1 for word in query_words if len(word) > 2 and word in fact.subject.lower())
        object_matches = sum(1 for word in query_words if len(word) > 2 and word in fact.object.lower())
        predicate_matches = sum(1 for word in query_words if len(word) > 2 and word in fact.predicate.lower())
        
        score += subject_matches * 8   # Subject matches weighted highest
        score += object_matches * 6    # Object matches weighted medium
        score += predicate_matches * 4 # Predicate matches weighted lowest

        # Semantic similarity bonus (domain-agnostic)
        subject_sim = calculate_semantic_similarity(query_lower, fact.subject.lower())
        object_sim = calculate_semantic_similarity(query_lower, fact.object.lower())
        predicate_sim = calculate_semantic_similarity(query_lower, fact.predicate.lower())
        
        score += subject_sim * 10
        score += object_sim * 8
        score += predicate_sim * 6

        # Confidence bonus (domain-agnostic)
        decayed_conf = getattr(fact, "decayed_confidence", 1.0)
        score += decayed_conf * 3

        # Recency bonus (more recent facts get slight boost)
        try:
            from datetime import datetime

            fact_time = datetime.fromisoformat(fact.timestamp.replace("Z", "+00:00"))
            now = datetime.now().replace(tzinfo=fact_time.tzinfo)
            days_old = (now - fact_time).days
            recency_bonus = max(
                0, 5 - (days_old * 0.1)
            )  # Up to 5 points for very recent facts
            score += recency_bonus
        except Exception:
            pass  # Ignore timestamp parsing errors

        # Contradiction penalty (domain-agnostic)
        if fact.contradiction_score > VOLATILITY_THRESHOLDS["high"]:
            score -= fact.contradiction_score * 5

        scored_facts.append((score, fact))

    # Filter out irrelevant facts (score < 5)
    relevant_facts = [fact for score, fact in scored_facts if score >= 5]

    # Apply weighted recency + confidence ranking to the relevant facts
    if relevant_facts:
        relevant_facts = rank_facts(relevant_facts)

    # Return top facts up to max_facts
    return relevant_facts[:max_facts]


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two text strings using Mistral embeddings.
    Falls back to word overlap if embeddings fail.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Quick check for identical text
    if text1.lower().strip() == text2.lower().strip():
        return 1.0
    
    try:
        # Use Mistral embeddings for semantic similarity
        embedding1 = get_cached_embedding(text1)
        embedding2 = get_cached_embedding(text2)
        
        # Calculate cosine similarity
        if np.any(embedding1) and np.any(embedding2):
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                # Ensure similarity is between 0 and 1
                return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        
    except Exception as e:
        logging.warning(f"Failed to calculate embedding-based similarity: {e}")
    
    # Fallback to word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def llm_confirms_contradiction(
    fact1: Tuple[str, str, str], fact2: Tuple[str, str, str]
) -> bool:
    """
    Use LLM to confirm if two facts are contradictory.

    Args:
        fact1: Tuple of (subject, predicate, object)
        fact2: Tuple of (subject, predicate, object)

    Returns:
        True if LLM confirms contradiction, False otherwise
    """
    try:
        import requests

        prompt = f"""Are these facts contradictory?
1. {fact1[0]} {fact1[1]} {fact1[2]}
2. {fact2[0]} {fact2[1]} {fact2[2]}

Answer only 'yes' or 'no':"""

        # Get Ollama host from config
        from config.settings import ollama_host
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=10,
        )
        response.raise_for_status()

        result = response.json()["response"].lower().strip()
        return "yes" in result

    except Exception as e:
        logging.warning(f"Warning: LLM contradiction check failed: {e}")
        return False  # Default to no contradiction if LLM fails


def _extract_subject_hybrid_confidence(
    text: str, fact_context: List[tuple] = None
) -> Tuple[str, float]:
    """
    LLM-driven subject extraction with confidence scoring.
    Uses Mistral to identify the main subject and estimate confidence.

    Args:
        text: Text to extract subject from
        fact_context: List of (subject, predicate, object) tuples from the database

    Returns:
        Tuple of (extracted_subject, confidence_score)
    """
    if not text:
        return "", 0.0

    original_text = text
    text = text.lower()

    # Remove brackets and their contents
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()

    # Remove common prefixes using config
    prefixes = _memory_config.get('normalization', {}).get('prefixes', [
        "your ", "my ", "his ", "her ", "their ", "the ", "a ", "an ", "this ", "that "
    ])
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]

    # Remove trailing qualifiers using config
    suffixes = _memory_config.get('normalization', {}).get('suffixes', [
        "now", "today", "always", "never"
    ])
    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[:-len(suffix)]

    text = text.strip()

    # If text is empty after cleaning, return empty
    if not text:
        return "", 0.0

    # Strategy 1: Use LLM for subject extraction
    cache_key = f"subject_extract:{text}"
    
    # Get prompt template from config
    prompt_template = _memory_config.get('normalization', {}).get('llm_prompts', {}).get(
        'subject_extraction', 
        "Extract the main subject from this text: '{text}'. Return only the subject word and a confidence score (0.0-1.0) separated by a comma. Example: 'pizza,0.9'"
    )
    
    prompt = prompt_template.format(text=text)
    
    # Call LLM
    response = _call_ollama_llm(prompt, cache_key)
    
    if response:
        try:
            # Parse "subject,confidence" format - handle quotes and extra text
            response_clean = response.strip().strip('"\'')
            if ',' in response_clean:
                subject_part, confidence_part = response_clean.split(',', 1)
                subject = subject_part.strip().strip('"\'')
                confidence_str = confidence_part.strip().strip('"\'')
                
                # Extract just the number from confidence using regex
                confidence_match = re.search(r'(\d+\.?\d*)', confidence_str)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    # Validate confidence range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    if subject:
                        return subject, confidence
        except (ValueError, AttributeError) as e:
            # Fallback if parsing fails
            logging.warning(f"Failed to parse LLM subject extraction response '{response}': {e}")
    
    # Strategy 2: Fallback for single words using basic heuristics
    if len(text.split()) == 1:
        word = text
        # Basic pronoun check
        if word in ["you", "i", "me", "he", "she", "they", "we", "us", "them", "it"]:
            return word, CONFIDENCE_THRESHOLDS["high"]
        else:
            # Use LLM to check if it's a noun
            noun_check_key = f"noun_check:{word}"
            noun_prompt = _memory_config.get('normalization', {}).get('llm_prompts', {}).get(
                'noun_check', "Is '{word}' a noun? Answer only 'yes' or 'no'."
            ).format(word=word)
            
            noun_response = _call_ollama_llm(noun_prompt, noun_check_key)
            
            if noun_response.lower().strip() == 'yes':
                return word, CONFIDENCE_THRESHOLDS["high"]
            else:
                return word, CONFIDENCE_THRESHOLDS["medium"]

    # Strategy 3: Multi-word fallback - use simpler heuristics
    words = text.split()
    if len(words) >= 2:
        # Remove articles and possessive pronouns to find main noun
        filtered_words = []
        skip_words = set(prefixes + suffixes + ["a", "an", "the"])
        
        for word in words:
            if word not in skip_words:
                filtered_words.append(word)
        
        if filtered_words:
            # Use the first meaningful word as subject
            return filtered_words[0], CONFIDENCE_THRESHOLDS["medium"]

    # Strategy 4: Context-based learning (if fact_context available)
    if fact_context:
        best_match = _learn_from_context(text, fact_context)
        if best_match:
            return best_match, CONFIDENCE_THRESHOLDS["high"]

    # Strategy 5: Default fallback
    return text, CONFIDENCE_THRESHOLDS["low"]


def _learn_from_context(text: str, fact_context: List[tuple]) -> str:
    """
    Learn subject patterns from existing facts in the database.
    """
    words = text.split()

    # Look for similar patterns in existing facts
    for existing_subject, existing_predicate, existing_object in fact_context:
        existing_words = existing_subject.lower().split()

        # If we have similar word patterns, use the same extraction logic
        if len(existing_words) >= 2 and len(words) >= 2:
            # Check if first word is the same (likely a verb)
            if existing_words[0] == words[0]:
                # Use the same extraction pattern
                if len(existing_words) >= 2:
                    return existing_words[1]  # Take second word as subject
                elif len(words) >= 2:
                    return words[1]

    return None


def _smart_forget_subject(subject_name: str, facts: List[tuple]) -> List[tuple]:
    """
    Smart subject forgetting that tries multiple matching strategies.

    Args:
        subject_name: Subject name to match
        facts: List of (id, subject, predicate, object) tuples

    Returns:
        List of facts to delete
    """
    # Extract subject with confidence
    target_subject, confidence = _extract_subject_hybrid_confidence(subject_name, facts)

    facts_to_delete = []

    for fact_id, subject, predicate, object_val in facts:
        # Try multiple matching strategies

        # Strategy 1: Direct subject match
        fact_subject, fact_confidence = _extract_subject_hybrid_confidence(
            subject, facts
        )
        if fact_subject.lower() == target_subject.lower():
            facts_to_delete.append(
                (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    "subject_match",
                    fact_confidence,
                )
            )
            continue

        # Strategy 2: Object match (for cases like "forget pizza" matching "love pizza")
        if object_val and target_subject.lower() in object_val.lower():
            facts_to_delete.append(
                (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    "object_match",
                    CONFIDENCE_THRESHOLDS["high"],
                )
            )
            continue

        # Strategy 3: Predicate match (for cases like "forget hate" matching "hate pizza")
        if predicate and target_subject.lower() in predicate.lower():
            facts_to_delete.append(
                (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    "predicate_match",
                    CONFIDENCE_THRESHOLDS["medium"],
                )
            )
            continue

        # Strategy 4: Semantic similarity (if embeddings available)
        try:
            if (
                _semantic_similarity(target_subject, subject)
                > CONFIDENCE_THRESHOLDS["high"]
            ):
                facts_to_delete.append(
                    (
                        fact_id,
                        subject,
                        predicate,
                        object_val,
                        "semantic_match",
                        CONFIDENCE_THRESHOLDS["very_high"],
                    )
                )
                continue
        except Exception:
            pass  # Skip if embeddings not available

    return facts_to_delete


def _semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using embeddings.
    """
    try:
        from scripts.embedder import embed, most_similar

        # Get embeddings
        emb1 = embed(text1)
        emb2 = embed(text2)

        if np.all(emb1 == 0) or np.all(emb2 == 0):
            logging.warning(
                f"Warning: Embedding failed for semantic similarity between '{text1}' and '{text2}'."
            )

        # Calculate similarity
        similarities = most_similar(emb1, topk=1)
        if similarities:
            return similarities[0][1]
        return 0.0
    except Exception:
        logging.exception("Error in _semantic_similarity")
        return 0.0


def _are_synonyms(word1: str, word2: str) -> bool:
    """Check if two words are synonyms using LLM-driven semantic analysis"""
    if not word1 or not word2:
        return False
    
    # Normalize words
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    
    # Quick check for identical words
    if word1 == word2:
        return True
    
    # Create cache key
    cache_key = f"synonym:{word1}:{word2}"
    
    # Get prompt template from config
    prompt_template = _memory_config.get('normalization', {}).get('llm_prompts', {}).get(
        'synonym_check', "Are {word1} and {word2} synonyms (same meaning)? Answer ONLY: yes or no"
    )
    
    prompt = prompt_template.format(word1=word1, word2=word2)
    
    # Call LLM
    response = _call_ollama_llm(prompt, cache_key)
    
    # Parse response - handle variations and quotes
    response_clean = response.lower().strip().strip('"\'').strip('.')
    return 'yes' in response_clean and 'no' not in response_clean


def calculate_agreement_score(fact1, fact2) -> float:
    """
    Calculate agreement score between two facts (synonym detection).

    Args:
        fact1: First TripletFact or tuple
        fact2: Second TripletFact or tuple

    Returns:
        Agreement score between 0.0 and 1.0 (1.0 = perfect agreement)
    """
    # 🩹 Handle case where tuples are passed instead of TripletFact objects
    if isinstance(fact1, tuple):
        from storage.memory_log import TripletFact

        fact1 = TripletFact(
            id=0,
            subject=fact1[0],
            predicate=fact1[1],
            object=fact1[2],
            source_message_id=0,
            timestamp="",
            frequency=1,
        )
    if isinstance(fact2, tuple):
        from storage.memory_log import TripletFact

        fact2 = TripletFact(
            id=0,
            subject=fact2[0],
            predicate=fact2[1],
            object=fact2[2],
            source_message_id=0,
            timestamp="",
            frequency=1,
        )

    try:
        # Check for None or empty objects
        if not fact1.object or not fact2.object:
            return 0.0

        # Normalize subjects for comparison
        subj1 = _normalize_subject(fact1.subject)
        subj2 = _normalize_subject(fact2.subject)

        # If subjects don't match, no agreement
        if subj1 != subj2:
            return 0.0

        # Check predicates and objects for agreement
        pred1 = fact1.predicate.lower().strip()
        pred2 = fact2.predicate.lower().strip()
        obj1 = fact1.object.lower().strip()
        obj2 = fact2.object.lower().strip()

        # Case 1: Same predicate, same object (perfect agreement)
        if pred1 == pred2 and obj1 == obj2:
            return 1.0

        # Case 2: Same predicate, synonym objects
        if pred1 == pred2 and _are_synonyms(obj1, obj2):
            return CONFIDENCE_THRESHOLDS["very_high"]

        # Case 3: Synonym predicates, same object
        if _are_synonyms(pred1, pred2) and obj1 == obj2:
            return CONFIDENCE_THRESHOLDS["very_high"]

        # Case 4: Synonym predicates, synonym objects
        if _are_synonyms(pred1, pred2) and _are_synonyms(obj1, obj2):
            return CONFIDENCE_THRESHOLDS["high"]

        # Case 5: Use embedding similarity for semantic agreement
        pred_sim = cosine_similarity(
            get_cached_embedding(pred1), get_cached_embedding(pred2)
        )
        obj_sim = cosine_similarity(
            get_cached_embedding(obj1), get_cached_embedding(obj2)
        )

        # High similarity in both predicate and object = agreement
        if (
            pred_sim > CONFIDENCE_THRESHOLDS["high"]
            and obj_sim > CONFIDENCE_THRESHOLDS["high"]
        ):
            return (pred_sim + obj_sim) / 2

        return 0.0

    except Exception as e:
        logging.warning(f"Agreement detection failed, using fallback: {e}")
        return 0.0


def group_contradictions_by_subject(
    contradictions: List[Tuple[TripletFact, TripletFact, float]],
) -> Dict[str, List[Tuple[TripletFact, TripletFact, float]]]:
    """
    Group contradictions by subject for better analysis.

    Args:
        contradictions: List of (fact1, fact2, score) tuples

    Returns:
        Dictionary mapping subjects to their contradictions
    """
    grouped = defaultdict(list)

    for fact1, fact2, score in contradictions:
        # Use the normalized subject for grouping
        subject = _normalize_subject(fact1.subject)
        grouped[subject].append((fact1, fact2, score))

    return dict(grouped)


def analyze_contradiction_clusters(
    contradictions: List[Tuple[TripletFact, TripletFact, float]],
) -> Dict[str, Dict]:
    """
    Analyze contradiction clusters to identify patterns and severity.

    Args:
        contradictions: List of (fact1, fact2, score) tuples

    Returns:
        Dictionary with cluster analysis for each subject
    """
    grouped = group_contradictions_by_subject(contradictions)
    analysis = {}

    for subject, subject_contradictions in grouped.items():
        # Calculate cluster statistics
        scores = [score for _, _, score in subject_contradictions]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        count = len(subject_contradictions)

        # Determine severity level
        if max_score >= CONFIDENCE_THRESHOLDS["high"]:
            severity = "CRITICAL"
        elif max_score >= CONFIDENCE_THRESHOLDS["medium"]:
            severity = "HIGH"
        elif max_score >= CONFIDENCE_THRESHOLDS["low"]:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Identify common patterns
        patterns = []
        for fact1, fact2, score in subject_contradictions:
            if fact1.predicate == fact2.predicate:
                patterns.append(
                    f"Same predicate '{fact1.predicate}', different objects"
                )
            elif fact1.object == fact2.object:
                patterns.append(f"Same object '{fact1.object}', different predicates")
            else:
                patterns.append(f"Different predicates and objects")

        analysis[subject] = {
            "contradiction_count": count,
            "average_score": avg_score,
            "max_score": max_score,
            "severity": severity,
            "patterns": list(set(patterns)),  # Remove duplicates
            "contradictions": subject_contradictions,
        }

    return analysis


def get_contradiction_summary_by_clusters(
    contradictions: List[Tuple[TripletFact, TripletFact, float]],
) -> str:
    """
    Generate a human-readable summary of contradiction clusters.

    Args:
        contradictions: List of (fact1, fact2, score) tuples

    Returns:
        Formatted summary string
    """
    if not contradictions:
        return f"{CONFIDENCE_ICONS['success']} No contradictions detected."

    analysis = analyze_contradiction_clusters(contradictions)

    summary = f"{CONFIDENCE_ICONS['high']} CONTRADICTION CLUSTER ANALYSIS\n"
    summary += "=" * 80 + "\n"

    # Sort by severity and count
    sorted_subjects = sorted(
        analysis.items(),
        key=lambda x: (
            x[1]["severity"] != "CRITICAL",
            -x[1]["contradiction_count"],
            -x[1]["max_score"],
        ),
    )

    for subject, cluster_data in sorted_subjects:
        summary += f"\n{CONFIDENCE_ICONS['high']} Subject: '{subject}'\n"
        summary += f"   Severity: {cluster_data['severity']}\n"
        summary += f"   Contradictions: {cluster_data['contradiction_count']}\n"
        summary += f"   Max Score: {cluster_data['max_score']:.3f}\n"
        summary += f"   Avg Score: {cluster_data['average_score']:.3f}\n"

        if cluster_data["patterns"]:
            summary += f"   Patterns: {', '.join(cluster_data['patterns'])}\n"

        # Show top contradictions
        top_contradictions = sorted(
            cluster_data["contradictions"], key=lambda x: x[2], reverse=True
        )[:3]
        summary += "   Top Contradictions:\n"
        for i, (fact1, fact2, score) in enumerate(top_contradictions, 1):
            summary += f"     {i}. Score: {score:.3f}\n"
            summary += f"        A: {fact1.subject} {fact1.predicate} {fact1.object}\n"
            summary += f"        B: {fact2.subject} {fact2.predicate} {fact2.object}\n"

    summary += f"\n{CONFIDENCE_ICONS['high']} Summary: {len(analysis)} subjects with contradictions, {len(contradictions)} total contradictions"

    return summary


def format_contradiction_summary(
    fact1: TripletFact, fact2: TripletFact, contradiction_score: float
) -> str:
    """
    Format a clear contradiction summary for user display.

    Args:
        fact1: First conflicting fact
        fact2: Second conflicting fact
        contradiction_score: Calculated contradiction score

    Returns:
        Formatted contradiction summary
    """
    conf1 = getattr(fact1, "decayed_confidence", 1.0)
    conf2 = getattr(fact2, "decayed_confidence", 1.0)

    summary = f"\n{CONFIDENCE_ICONS['medium']} CONTRADICTION DETECTED (score: {contradiction_score:.2f}):\n"
    summary += f"A: '{fact1.subject} {fact1.predicate} {fact1.object}' (confidence: {conf1:.2f})\n"
    summary += f"B: '{fact2.subject} {fact2.predicate} {fact2.object}' (confidence: {conf2:.2f})\n"

    if contradiction_score > CONFIDENCE_THRESHOLDS["high"]:
        summary += f"{VOLATILITY_ICONS['high']} HIGH CONTRADICTION - Please clarify which is correct.\n"
    elif contradiction_score > CONFIDENCE_THRESHOLDS["medium"]:
        summary += f"{VOLATILITY_ICONS['medium']} MODERATE CONTRADICTION - These statements conflict.\n"
    else:
        summary += f"{VOLATILITY_ICONS['stable']} MINOR CONTRADICTION - Slight inconsistency detected.\n"

    return summary


def get_most_recent_resolved_fact(facts, subject, predicate=None):
    """
    Return the most recent, high-confidence, non-contradicted fact for a subject (and optional predicate).
    """
    from storage.memory_utils import _normalize_subject
    subject = _normalize_subject(subject)
    filtered = [
        f for f in facts
        if _normalize_subject(f.subject) == subject and (predicate is None or f.predicate == predicate)
        and (getattr(f, 'contradiction_score', 0.0) < 0.3)
    ]
    if not filtered:
        return None
    # Sort by confidence, then timestamp
    filtered.sort(key=lambda f: (-(getattr(f, 'decayed_confidence', f.confidence)), f.timestamp), reverse=True)
    return filtered[0]

# Store reference to original build_memory_context for fallback
old_build_memory_context = build_memory_context

# Add a simple semantic similarity function if not present
try:
    calculate_semantic_similarity
except NameError:
    def calculate_semantic_similarity(a, b):
        if not a or not b:
            return 0.0
        a = a.lower().strip()
        b = b.lower().strip()
        # Use SequenceMatcher for fuzzy match (can be replaced with embedding similarity if available)
        return SequenceMatcher(None, a, b).ratio()

def is_valid_fact(fact) -> bool:
    """
    Return True if fact is valid for use in prompts:
    - subject, predicate, object are all non-empty strings
    - confidence > low threshold
    - contradiction_score <= high threshold
    """
    from config.settings import CONFIDENCE_THRESHOLDS

    if (
        not hasattr(fact, "subject")
        or not hasattr(fact, "predicate")
        or not hasattr(fact, "object")
    ):
        return False
    if not fact.subject or not fact.predicate or not fact.object:
        return False
    if getattr(fact, "confidence", 1.0) <= CONFIDENCE_THRESHOLDS["low"]:
        return False
    if (
        getattr(fact, "contradiction_score", 0.0)
        > DEFAULT_VALUES["contradiction_threshold"]
    ):
        return False
    return True


def normalize_question_to_subject(question: str) -> str:
    """
    Normalize a question to its subject for fact recall.
    E.g., 'what color cats do I like' -> 'user', 'my favorite color' -> 'user'
    """
    if not question:
        return ''
    q = question.lower()
    pronouns = ['i', 'me', 'my', 'mine', 'myself']
    for pronoun in pronouns:
        if f' {pronoun} ' in f' {q} ' or q.startswith(pronoun + ' '):
            return 'user'
    # Fallback: try to extract a noun
    import re
    match = re.search(r'\b(\w+)\b', q)
    if match:
        return match.group(1)
    return q.strip()

def _is_question(text: str) -> bool:
    """
    Detect if text is a question using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        True if text appears to be a question
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Direct question indicators
    question_words = ["what", "when", "where", "who", "why", "how", "which", "whats", "whens", "wheres", "whos", "whys", "hows"]
    if any(text_lower.startswith(word) for word in question_words):
        return True
        
    # Question mark
    if "?" in text:
        return True
        
    # Common question patterns
    question_patterns = [
        r"^(is|are|was|were|am|do|does|did|can|could|will|would|should|may|might)\s+\w+",
        r"^(what|when|where|who|why|how|which)\s+\w+",
        r"^(whats|whens|wheres|whos|whys|hows)\s+\w+",
    ]
    
    for pattern in question_patterns:
        if re.match(pattern, text_lower):
            return True
            
    return False

def find_facts_for_question(question: str, memory_log) -> List[TripletFact]:
    """
    Find the most relevant facts for a question by normalizing the question to a subject
    and searching for matching facts.
    
    Args:
        question: Question text (e.g., "what's my favorite color?")
        memory_log: MemoryLog instance
        
    Returns:
        List of TripletFact objects ordered by confidence DESC, timestamp DESC
    """
    if not question:
        return []
    
    # Normalize question to subject
    normalized_subject = normalize_question_to_subject(question)
    
    if not normalized_subject:
        return []
    
    # Strategy 1: Try exact subject match first
    exact_facts = memory_log.get_facts_about(normalized_subject)
    if exact_facts:
        # Sort by confidence DESC, timestamp DESC
        exact_facts.sort(key=lambda f: (-f.confidence, f.timestamp), reverse=True)
        return exact_facts
    
    # Strategy 2: Try semantic similarity with existing subjects
    try:
        all_facts = memory_log.get_all_facts()
        if not all_facts:
            return []
        
        # Get all unique subjects
        subjects = list(set(f.subject for f in all_facts))
        
        # Find most similar subject using embeddings
        from scripts.embedder import embed
        import numpy as np
        
        question_embedding = embed(normalized_subject)
        if np.all(question_embedding == 0):
            return []
        
        best_match = None
        best_similarity = 0.0
        
        for subject in subjects:
            subject_embedding = embed(subject)
            if np.all(subject_embedding == 0):
                continue
            
            similarity = float(np.dot(question_embedding, subject_embedding))
            if similarity > best_similarity and similarity > CONFIDENCE_THRESHOLDS["medium"]:
                best_similarity = similarity
                best_match = subject
        
        if best_match:
            # Get facts for the best matching subject
            semantic_facts = memory_log.get_facts_about(best_match)
            semantic_facts.sort(key=lambda f: (-f.confidence, f.timestamp), reverse=True)
            return semantic_facts
            
    except Exception as e:
        logging.warning(f"Semantic search failed for question '{question}': {e}")
    
    # Strategy 3: Fallback to word-based matching
    question_words = set(normalized_subject.lower().split())
    
    word_matched_facts = []
    for fact in memory_log.get_all_facts():
        fact_words = set(fact.subject.lower().split())
        if question_words & fact_words:  # If there's any word overlap
            word_matched_facts.append(fact)
    
    if word_matched_facts:
        word_matched_facts.sort(key=lambda f: (-f.confidence, f.timestamp), reverse=True)
        return word_matched_facts
    
    return []

def normalize_predicate(predicate: str) -> str:
    """
    Normalize predicate to canonical form using synonym mapping.

    Args:
        predicate: Input predicate to normalize

    Returns:
        Canonical predicate form
    """
    # Defensive: handle None or empty predicate
    safe_pred = (predicate or "").lower().strip()
    if not safe_pred:
        return ""

    # Define synonym mapping inline since SYNONYM_MAP was removed
    synonym_map = {
        "love": "like",
        "adore": "like", 
        "enjoy": "like",
        "prefer": "like",
        "hate": "dislike",
        "loathe": "dislike",
        "despise": "dislike",
        "abhor": "dislike",
        "good": "great",
        "excellent": "great",
        "wonderful": "great",
        "amazing": "great",
        "terrible": "bad",
        "awful": "bad",
        "horrible": "bad",
        "dreadful": "bad",
    }
    
    # Direct synonym mapping
    if safe_pred in synonym_map:
        return synonym_map[safe_pred]

    # Try semantic similarity for close matches
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Load model (cache it)
        if not hasattr(normalize_predicate, "_model"):
            normalize_predicate._model = SentenceTransformer("all-MiniLM-L6-v2")

        model = normalize_predicate._model

        # Get embeddings for input and canonical predicates
        canonical_predicates = list(set(synonym_map.values()))
        canonical_embeddings = model.encode(canonical_predicates)
        input_embedding = model.encode([safe_pred])

        # Find most similar canonical predicate
        similarities = np.dot(canonical_embeddings, input_embedding.T).flatten()
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        # Only normalize if similarity is high enough
        if best_similarity > CONFIDENCE_THRESHOLDS["medium"]:
            return canonical_predicates[best_match_idx]

    except Exception:
        # Fallback to direct mapping only
        pass

    # Return original if no good match found
    return safe_pred


def get_sentiment_score(predicate: str) -> float:
    """
    Compute emotional valence of predicate using intensity mapping for known words,
    otherwise fall back to embedding-based similarity.
    """
    if not predicate:
        return 0.0

    # Configurable intensity map for human-intuitive ordering
    from config.environment import settings
    intensity_map = {
        "love": settings.sentiment_love_intensity,
        "like": settings.sentiment_like_intensity,
        "neutral": 0.0,
        "dislike": settings.sentiment_dislike_intensity,
        "hate": settings.sentiment_hate_intensity,
    }
    predicate_lower = predicate.lower().strip()
    if predicate_lower in intensity_map:
        return intensity_map[predicate_lower]

    try:
        predicate_embedding = get_cached_embedding(predicate_lower)
        positive_reference_words = ["love", "like", "adore", "enjoy", "prefer", "good", "great", "excellent", "wonderful", "amazing"]
        negative_reference_words = ["hate", "dislike", "loathe", "despise", "abhor", "terrible", "awful", "horrible", "dreadful"]
        positive_embeddings = [get_cached_embedding(pos) for pos in positive_reference_words]
        negative_embeddings = [get_cached_embedding(neg) for neg in negative_reference_words]
        positive_similarities = [cosine_similarity(predicate_embedding, pos_emb) for pos_emb in positive_embeddings]
        max_positive_sim = max(positive_similarities) if positive_similarities else 0.0
        negative_similarities = [cosine_similarity(predicate_embedding, neg_emb) for neg_emb in negative_embeddings]
        max_negative_sim = max(negative_similarities) if negative_similarities else 0.0
        sentiment_score = max_positive_sim - max_negative_sim
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        return sentiment_score
    except Exception as e:
        logging.warning(f"Error computing sentiment score for '{predicate}': {e}")
        return 0.0


def get_volatility_score(facts: List[TripletFact]) -> float:
    """
    Detect contradictory emotional flips over time based on sentiment changes.
    Uses dynamic thresholds for volatility calculation.

    Args:
        facts: List of TripletFact objects for a subject/object pair

    Returns:
        Volatility score from 0.0 (stable) to 1.0 (highly volatile)
    """
    if len(facts) < 2:
        return 0.0

    try:
        # Sort facts by timestamp
        sorted_facts = sorted(facts, key=lambda f: f.timestamp)

        # Calculate sentiment scores for each fact
        sentiment_scores = []
        for fact in sorted_facts:
            sentiment = get_sentiment_score(fact.predicate)
            sentiment_scores.append(sentiment)

        # Count sign changes (emotional flips)
        sign_changes = 0
        for i in range(1, len(sentiment_scores)):
            prev_sign = (
                1
                if sentiment_scores[i - 1] > 0
                else (-1 if sentiment_scores[i - 1] < 0 else 0)
            )
            curr_sign = (
                1 if sentiment_scores[i] > 0 else (-1 if sentiment_scores[i] < 0 else 0)
            )

            if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
                sign_changes += 1

        # Calculate volatility as ratio of sign changes to total transitions
        total_transitions = len(sentiment_scores) - 1
        if total_transitions == 0:
            return 0.0

        volatility = sign_changes / total_transitions

        # Boost volatility if there are large sentiment swings (dynamic threshold)
        sentiment_range = max(sentiment_scores) - min(sentiment_scores)
        sentiment_range_threshold = PERSONALITY_ADAPTATION["sentiment_range_threshold"]
        volatility_boost_factor = PERSONALITY_ADAPTATION["volatility_boost_factor"]

        if (
            sentiment_range > sentiment_range_threshold
        ):  # Large swing from very negative to very positive
            volatility = min(1.0, volatility * volatility_boost_factor)

        return volatility

    except Exception as e:
        logging.warning(f"Error computing volatility score: {e}")
        return 0.0


def get_sentiment_trajectory(facts: List[TripletFact]) -> Dict[str, float]:
    """
    Compute sentiment trajectory (emotional arc) for a set of facts.

    Args:
        facts: List of TripletFact objects for a subject/object pair

    Returns:
        Dictionary with trajectory metrics:
        - slope: Linear regression slope (positive = improving sentiment)
        - intercept: Y-intercept of regression line
        - r_squared: Goodness of fit
        - recent_sentiment: Average sentiment of last 3 facts
        - volatility: Volatility score
    """
    if len(facts) < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "recent_sentiment": 0.0,
            "volatility": 0.0,
        }

    try:
        # Sort facts by timestamp
        sorted_facts = sorted(facts, key=lambda f: f.timestamp)

        # Convert timestamps to days since first fact
        first_timestamp = sorted_facts[0].timestamp
        try:
            first_date = datetime.strptime(first_timestamp.split()[0], "%Y-%m-%d")
        except Exception:
            # Fallback: use fact index as time proxy
            first_date = datetime.now()

        time_points = []
        sentiment_points = []

        for i, fact in enumerate(sorted_facts):
            try:
                fact_date = datetime.strptime(fact.timestamp.split()[0], "%Y-%m-%d")
                days_since = (fact_date - first_date).days
            except Exception:
                # Fallback: use index as time proxy
                days_since = i

            sentiment = get_sentiment_score(fact.predicate)
            time_points.append(days_since)
            sentiment_points.append(sentiment)

        # Log input details for debugging
        logger = logging.getLogger(__name__)
        logger.debug(f"Sentiment trajectory input - shape: {len(time_points)}x{len(sentiment_points)}, "
                    f"time_points: {time_points}, sentiment_points: {sentiment_points}")

        # Convert to numpy arrays for matrix operations
        X = np.array(time_points).reshape(-1, 1)
        y = np.array(sentiment_points)

        # Defensive checks for matrix validity
        if X.shape[0] < 2:
            logger.warning("Invalid sentiment matrix — embedding matrix has < 2 rows, skipping trajectory computation.")
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "recent_sentiment": np.mean(sentiment_points) if sentiment_points else 0.0,
                "volatility": 0.0,
            }

        # Check if all rows are equal (no variation in time)
        if np.all(X == X[0]):
            logger.warning("Invalid sentiment matrix — all time points are equal, skipping trajectory computation.")
            return {
                "slope": 0.0,
                "intercept": np.mean(sentiment_points) if sentiment_points else 0.0,
                "r_squared": 0.0,
                "recent_sentiment": np.mean(sentiment_points) if sentiment_points else 0.0,
                "volatility": 0.0,
            }

        # Check for NaN or inf values
        if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any():
            logger.warning("Invalid sentiment matrix — contains NaN or inf values, skipping trajectory computation.")
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "recent_sentiment": 0.0,
                "volatility": 0.0,
            }

        # Check matrix rank for linear independence (need at least rank 1 for 1D regression)
        if np.linalg.matrix_rank(X) < 1:
            logger.warning("Invalid sentiment matrix — insufficient rank for linear regression, skipping trajectory computation.")
            return {
                "slope": 0.0,
                "intercept": np.mean(sentiment_points) if sentiment_points else 0.0,
                "r_squared": 0.0,
                "recent_sentiment": np.mean(sentiment_points) if sentiment_points else 0.0,
                "volatility": 0.0,
            }

        # Compute linear regression with error handling
        try:
            coeffs = np.polyfit(time_points, sentiment_points, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # Calculate R-squared
            y_pred = np.polyval(coeffs, time_points)
            ss_res = np.sum((sentiment_points - y_pred) ** 2)
            ss_tot = np.sum((sentiment_points - np.mean(sentiment_points)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        except np.linalg.LinAlgError as e:
            logger.warning(f"Linear algebra error in sentiment trajectory computation: {e}. Retrying with regularization.")
            
            # Retry with regularization (add small noise to break degeneracy)
            try:
                # Add small regularization to time points
                regularized_time = np.array(time_points) + np.random.normal(0, 1e-6, len(time_points))
                coeffs = np.polyfit(regularized_time, sentiment_points, 1)
                slope = coeffs[0]
                intercept = coeffs[1]

                # Calculate R-squared with regularized data
                y_pred = np.polyval(coeffs, regularized_time)
                ss_res = np.sum((sentiment_points - y_pred) ** 2)
                ss_tot = np.sum((sentiment_points - np.mean(sentiment_points)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                
                logger.info("Successfully computed sentiment trajectory with regularization.")
                
            except Exception as reg_error:
                logger.error(f"Failed to compute sentiment trajectory even with regularization: {reg_error}")
                slope = 0.0
                intercept = np.mean(sentiment_points) if sentiment_points else 0.0
                r_squared = 0.0

        # Calculate recent sentiment (last 3 facts)
        recent_sentiment = (
            np.mean(sentiment_points[-3:])
            if len(sentiment_points) >= 3
            else np.mean(sentiment_points)
        )

        # Calculate volatility
        volatility = get_volatility_score(sorted_facts)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "recent_sentiment": recent_sentiment,
            "volatility": volatility,
        }

    except Exception as e:
        logging.warning(f"Error computing sentiment trajectory: {e}")
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "recent_sentiment": 0.0,
            "volatility": 0.0,
        }


def compute_decay_weighted_confidence(
    confidence: float,
    timestamp: str,
    decay_rate: float = DEFAULT_VALUES["decay_rate_default"],
) -> float:
    """
    Compute recency-weighted confidence that prioritizes recent facts.

    Args:
        confidence: Base confidence score (0.0 to 1.0)
        timestamp: Timestamp string in format "YYYY-MM-DD HH:MM:SS"
        decay_rate: Daily decay rate (default = 2% decay per day)

    Returns:
        Decay-weighted confidence score
    """
    try:
        # Parse timestamp
        if isinstance(timestamp, str):
            # Handle different timestamp formats
            if " " in timestamp:
                fact_date = datetime.strptime(timestamp.split()[0], "%Y-%m-%d")
            else:
                fact_date = datetime.strptime(timestamp, "%Y-%m-%d")
        else:
            # If timestamp is already a datetime object
            fact_date = timestamp

        # Calculate days since the fact was created
        current_date = datetime.now()
        days_since = (current_date - fact_date).days

        # Apply exponential decay
        decay_factor = decay_rate**days_since

        # Weight confidence by recency
        weighted_confidence = confidence * decay_factor

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, weighted_confidence))

    except Exception as e:
        logging.warning(f"Error computing decay-weighted confidence: {e}")
        return confidence  # Return original confidence as fallback


def rank_facts(facts: List[TripletFact]) -> List[TripletFact]:
    """
    Rank facts by weighted recency + confidence instead of confidence only.
    
    Args:
        facts: List of TripletFact objects to rank
        
    Returns:
        List of TripletFact objects sorted by weighted score (highest first)
    """
    if not facts:
        return []
    
    def score(fact):
        # Get timestamp as epoch time for recency calculation
        try:
            from datetime import datetime
            # Parse timestamp - handle different formats
            if hasattr(fact, 'timestamp') and fact.timestamp:
                if isinstance(fact.timestamp, str):
                    # Try ISO format first
                    try:
                        fact_time = datetime.fromisoformat(fact.timestamp.replace("Z", "+00:00"))
                    except ValueError:
                        # Try other common formats
                        try:
                            fact_time = datetime.strptime(fact.timestamp, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            # Fallback to current time if parsing fails
                            fact_time = datetime.now()
                else:
                    fact_time = datetime.now()
            else:
                fact_time = datetime.now()
            
            # Convert to epoch timestamp
            fact_epoch = fact_time.timestamp()
        except Exception:
            # Fallback to current time if timestamp parsing fails
            fact_epoch = datetime.now().timestamp()
        
        # Get confidence score (default to 1.0 if not available)
        confidence = getattr(fact, 'confidence', 1.0)
        
        # Calculate recency and confidence weights
        recency_weight = 0.7
        confidence_weight = 0.3
        
        # Normalize time to [0, 1] range across all facts
        timestamps = []
        for f in facts:
            try:
                if hasattr(f, 'timestamp') and f.timestamp:
                    if isinstance(f.timestamp, str):
                        try:
                            t = datetime.fromisoformat(f.timestamp.replace("Z", "+00:00"))
                        except ValueError:
                            try:
                                t = datetime.strptime(f.timestamp, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                t = datetime.now()
                    else:
                        t = datetime.now()
                else:
                    t = datetime.now()
                timestamps.append(t.timestamp())
            except Exception:
                timestamps.append(datetime.now().timestamp())
        
        if timestamps:
            oldest_ts = min(timestamps)
            newest_ts = max(timestamps)
            time_range = newest_ts - oldest_ts
            
            if time_range > 0:
                normalized_time = (fact_epoch - oldest_ts) / time_range
            else:
                normalized_time = 0.5  # If all facts have same timestamp
        else:
            normalized_time = 0.5
        
        # Calculate weighted score
        weighted_score = recency_weight * normalized_time + confidence_weight * confidence
        
        return weighted_score
    
    # Sort facts by weighted score (highest first)
    return sorted(facts, key=score, reverse=True)


def find_facts_for_question(question: str, memory_log) -> List[TripletFact]:
    """
    Find the most relevant facts for a question by normalizing the question to a subject
    and searching for matching facts.
    
    Args:
        question: Question text (e.g., "what's my favorite color?")
        memory_log: MemoryLog instance
        
    Returns:
        List of TripletFact objects ordered by weighted recency + confidence
    """
    if not question:
        return []
    
    # Normalize question to subject
    normalized_subject = normalize_question_to_subject(question)
    
    if not normalized_subject:
        return []
    
    # Strategy 1: Try exact subject match first
    exact_facts = memory_log.get_facts_about(normalized_subject)
    if exact_facts:
        # Use weighted recency + confidence ranking instead of confidence only
        return rank_facts(exact_facts)
    
    # Strategy 2: Try semantic similarity with existing subjects
    try:
        all_facts = memory_log.get_all_facts()
        if not all_facts:
            return []
        
        # Get all unique subjects
        subjects = list(set(f.subject for f in all_facts))
        
        # Find most similar subject using embeddings
        from scripts.embedder import embed
        import numpy as np
        
        question_embedding = embed(normalized_subject)
        if np.all(question_embedding == 0):
            return []
        
        best_match = None
        best_similarity = 0.0
        
        for subject in subjects:
            subject_embedding = embed(subject)
            if np.all(subject_embedding == 0):
                continue
            
            similarity = float(np.dot(question_embedding, subject_embedding))
            if similarity > best_similarity and similarity > CONFIDENCE_THRESHOLDS["medium"]:
                best_similarity = similarity
                best_match = subject
        
        if best_match:
            # Get facts for the best matching subject
            semantic_facts = memory_log.get_facts_about(best_match)
            return rank_facts(semantic_facts)
            
    except Exception as e:
        logging.warning(f"Semantic search failed for question '{question}': {e}")
    
    # Strategy 3: Fallback to word-based matching
    question_words = set(normalized_subject.lower().split())
    
    word_matched_facts = []
    for fact in memory_log.get_all_facts():
        fact_words = set(fact.subject.lower().split())
        if question_words & fact_words:  # If there's any word overlap
            word_matched_facts.append(fact)
    
    if word_matched_facts:
        return rank_facts(word_matched_facts)
    
    return []

# --- PATCH: Plural recall and robust contradiction resolution ---
import re

def is_plural_query(query: str) -> bool:
    """Detect if a query is plural (e.g., 'what are', 'list', or ends with 's')."""
    q = query.lower().strip()
    if re.search(r'\b(are|list|which|what are|what kinds|what types|what colors|favorites|preferences)\b', q):
        return True
    # Heuristic: ends with 's' and not a possessive
    if q.endswith('s') and not q.endswith("'s"):
        return True
    return False

# Patch build_smart_memory_context to support plural recall and robust contradiction resolution
def build_smart_memory_context(triplets, user_query, max_tokens=512, memory_log=None):
    import re
    import time
    from storage.memory_utils import calculate_semantic_similarity
    
    # Enhanced memory context generation with compound query support
    
    preference_patterns = [
        r'what.*(do i|does user|my|mine).*like',
        r'what do i like',  # Direct match for simple queries
        r'what.*i like',    # Match "what do i like", "what things i like" 
        r'what.*cats.*do.*i.*like',  # NEW: Match "what color cats do i like"
        r'which.*(do i|does user|my|mine).*prefer',
        r'fav(ourite|orite)?',  # match 'fav', 'favourite', 'favorite'
        r'favorite',
        r'prefer',
        r'like best',
        r'like most',
        r'what.*color',
        r'what.*number',
        r'what.*food',
        r'what.*animal',
    ]
    is_preference = any(re.search(p, user_query.lower()) for p in preference_patterns)
    
    # Enhanced preference detection for compound queries
    
    if is_preference:
        # ENHANCED: Handle "what color X do i like" queries with intelligent extraction
        if 'color' in user_query.lower() and any(word in user_query.lower() for word in ['fav', 'favorite', 'like', 'do i like']):
            # Extract the object (e.g., "cat" from "what color cats do i like")
            words = user_query.lower().split()
            content_words = [w for w in words if w not in {'what', 'whats', "what's", 'do', 'i', 'does', 'user', 'my', 'mine', 'like', 'prefer', 'which', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'with', 'most', 'best', 'kind', 'type', 'sort', 'sorts', 'are', 'list', 'now', 'currently', 'currently,', 'fav', 'favorite', 'color'}]
            
            if content_words:
                target_object = content_words[0]  # e.g., "cat"
                
                # Look for facts where user likes [color] [object]
                for fact in triplets:
                    if fact.predicate.lower() == 'like' and target_object in fact.object.lower():
                        # Extract color from "color object" pattern (e.g., "orange cats")
                        obj_words = fact.object.lower().split()
                        
                        # Handle singular/plural matching (e.g., "cat" matches "cats")
                        target_index = -1
                        for i, word in enumerate(obj_words):
                            if word == target_object or word == target_object + 's' or (word.endswith('s') and word[:-1] == target_object):
                                target_index = i
                                break
                        
                        if target_index >= 0 and target_index > 0:  # There's a word before the object
                            color_word = obj_words[target_index - 1]
                            return f"Your favorite {target_object} color is {color_word} (confidence: {getattr(fact, 'confidence', 1.0):.2f})"
        
        # FIXED: Improved query parsing for compound queries like "what color doors do i like"
        words = user_query.lower().split()
        skip_words = {'what', 'whats', "what's", 'do', 'i', 'does', 'user', 'my', 'mine', 'like', 'prefer', 'which', 'is', 'the', 'a', 'an', 'of', 'to', 'in', 'on', 'for', 'with', 'most', 'best', 'kind', 'type', 'sort', 'sorts', 'are', 'list', 'now', 'currently', 'currently,', 'fav', 'favorite'}
        
        # Extract non-skip words in order
        content_words = [w for w in words if w not in skip_words]
        print(f"[QUERY PARSING] content_words: {content_words}")
        
        # COMPOUND QUERY PARSING: Handle "what [property] [object]" patterns
        # Examples: "what color dogs", "what color cars", "what size boats"
        # Also handle "fav X of Y" patterns like "fav color of dog"
        main_object = None
        query_property = None
        
        # Special handling for "fav X of Y" patterns first
        print(f"[QUERY PARSING] Checking 'fav X of Y' pattern...")
        if 'of' in user_query.lower() and any(word in user_query.lower() for word in ['fav', 'favorite']):
            # Pattern: "whats my fav color of dog" → main_object="dog", query_property="color"
            of_index = words.index('of') if 'of' in words else -1
            if of_index > 0 and of_index < len(words) - 1:
                # Get the word after "of" as the main object
                main_object = words[of_index + 1]
                # Get the property word before "of" 
                for i in range(of_index - 1, -1, -1):
                    if words[i] not in skip_words:
                        query_property = words[i]
                        break
                logging.info(f"[COMPOUND QUERY] Detected 'fav X of Y' pattern: property='{query_property}', object='{main_object}'")
        
        # FIXED: Special handling for "what is my favorite [property]" patterns (e.g., "what is my favorite color")
        elif any(word in user_query.lower() for word in ['fav', 'favorite']) and len(content_words) >= 1:
            # Pattern: "what is my favorite color" → query_property="color"
            # Pattern: "what is my fav cat color" → main_object="cat", query_property="color"
            property_words = {'color', 'size', 'type', 'kind', 'style', 'brand', 'material', 'food', 'animal', 'fruit', 'number'}
            
            # ENHANCED: Handle both "fav [property]" and "fav [object] [property]" patterns
            if len(content_words) >= 2:
                # Check if we have [object] [property] pattern
                potential_object = content_words[0]
                potential_property = content_words[1]
                
                if potential_property in property_words:
                    main_object = potential_object
                    query_property = potential_property
                    logging.info(f"[COMPOUND QUERY] Detected 'fav [object] [property]' pattern: object='{main_object}', property='{query_property}'")
                elif potential_object in property_words:
                    # Check if first word is a property (e.g., "fav color")
                    query_property = potential_object
                    main_object = None
                    print(f"[FAVORITE QUERY] Detected 'fav [property]' pattern: property='{query_property}'")
            elif len(content_words) == 1:
                # Single word after "fav" - check if it's a property
                word = content_words[0]
                if word in property_words:
                    query_property = word
                    main_object = None
                    print(f"[FAVORITE QUERY] Detected 'favorite [property]' pattern: property='{query_property}'")
        
        # If not a "fav X of Y" pattern, try standard property-object parsing
        print(f"[QUERY PARSING] After first parsing attempt: main_object='{main_object}', query_property='{query_property}'")
        if not main_object and not query_property and len(content_words) >= 2:
            # Check for property-object pattern (e.g., "color doors")
            potential_property = content_words[0]
            potential_object = content_words[1]
            
            # Properties that modify objects
            property_words = {'color', 'size', 'type', 'kind', 'style', 'brand', 'material'}
            
            if potential_property in property_words:
                # This is a "what [property] [object]" query
                query_property = potential_property
                main_object = potential_object
                logging.info(f"[Compound Query] Detected property-object query: property='{query_property}', object='{main_object}'")
            elif potential_object in property_words:
                # This is a "what [object] [property]" query (e.g., "what cat color")
                query_property = potential_object
                main_object = potential_property
                logging.info(f"[Compound Query] Detected object-property query: property='{query_property}', object='{main_object}'")
            else:
                # Traditional subject extraction - for ambiguous cases, use first word as main object
                main_object = potential_property  # Use first word as main object
                query_property = potential_object  # Second word as property/context
        elif len(content_words) == 1:
            main_object = content_words[0]
            query_property = None
        else:
            # GENERAL PREFERENCE QUERY: No specific content words (e.g., "what do i like")
            main_object = None
            query_property = None
        
        # Special handling for general preference queries  
        # Check for general preference queries
        
        # CRITICAL FIX: Handle compound color+object queries directly  
        if main_object and query_property == 'color':
            # Enhanced compound color query processing
            # Look for facts where object contains the main_object
            for fact in triplets:
                if fact.predicate.lower() == 'like' and main_object in fact.object.lower():
                    # Extract color from the object (e.g., "orange cats" -> "orange")
                    words = fact.object.lower().split()
                    if main_object in words:
                        object_index = words.index(main_object)
                        if object_index > 0:  # There's a word before the object
                            color_word = words[object_index - 1]
                            return f"Your favorite {main_object} color is {color_word} (confidence: {getattr(fact, 'confidence', 1.0):.2f})"
        
        if not main_object and not query_property:
            # For queries like "what do i like" or "what colors do i like", look for any "user like *" facts
            logging.info(f"[General Preference Query] No specific content words, searching for general preferences")
            general_facts = [f for f in triplets if f.subject.lower() == 'user' and f.predicate.lower() == 'like']
            
            # Check if this is a plural query that should return multiple results
            is_plural = any(word in user_query.lower() for word in ['colors', 'things', 'items', 'stuff', 'preferences']) or not ('favorite' in user_query.lower() or 'most' in user_query.lower())
            
            if general_facts:
                if 'most' in user_query.lower():
                    sorted_facts = sorted(general_facts, key=lambda f: getattr(f, 'decayed_confidence', f.confidence), reverse=True)
                    best_fact = sorted_facts[0]
                    return f"You like {best_fact.object} the most (confidence: {getattr(best_fact, 'decayed_confidence', best_fact.confidence):.2f})"
                elif 'least' in user_query.lower():
                    sorted_facts = sorted(general_facts, key=lambda f: getattr(f, 'decayed_confidence', f.confidence))
                    worst_fact = sorted_facts[0]
                    return f"You like {worst_fact.object} the least (confidence: {getattr(worst_fact, 'decayed_confidence', worst_fact.confidence):.2f})"
                elif is_plural and len(general_facts) > 1:
                    # Return multiple preferences for plural queries
                    sorted_facts = sorted(general_facts, key=lambda f: getattr(f, 'decayed_confidence', f.confidence), reverse=True)
                    preferences = [f.object for f in sorted_facts[:3]]  # Top 3 preferences
                    logging.info(f"[Plural Preference Query] Found {len(preferences)} preferences: {preferences}")
                    return f"You like: {', '.join(preferences)}"
                else:
                    # Return the most recent/confident fact for singular queries
                    best_fact = max(general_facts, key=lambda f: getattr(f, 'decayed_confidence', f.confidence))
                    logging.info(f"[General Preference Query] Found general preference: {best_fact.subject} {best_fact.predicate} {best_fact.object}")
                    return f"user {best_fact.predicate} {best_fact.object} (confidence: {getattr(best_fact, 'decayed_confidence', best_fact.confidence):.2f})"
            else:
                logging.info(f"[General Preference Query] No general preferences found")
                return "I couldn't find any preferences in your memory. Please tell me what you like!"
        
        print(f"[PARSED QUERY] '{user_query}' -> main_object='{main_object}', query_property='{query_property}', content_words={content_words}")
        
        # Find best matching fact using semantic similarity
        best_score = 0.0
        best_fact = None
        candidates = []
        
        # Special handling for color and property queries
        is_color_query = query_property == 'color' or 'color' in user_query.lower()
        is_property_query = query_property in {'color', 'size', 'type', 'kind', 'style', 'brand', 'material', 'food', 'animal', 'fruit', 'number'}
        is_favorite_query = 'favorite' in user_query.lower() or 'fav' in user_query.lower()
        
        print(f"[MATCHING SETUP] is_color_query={is_color_query}, is_property_query={is_property_query}, is_favorite_query={is_favorite_query}")
        print(f"[MATCHING SETUP] Starting fact matching with {len(triplets)} triplets...")
        
        # SPECIAL CASE: Direct favorite property matching (e.g., "what is my favorite color" → "user favorite color")
        print(f"[SPECIAL CASE CHECK] is_favorite_query={is_favorite_query}, query_property='{query_property}', main_object='{main_object}'")
        if is_favorite_query and query_property and not main_object:
            # Look for facts where the subject contains both "favorite" and the property
            for fact in triplets:
                subject_lower = fact.subject.lower()
                if ('favorite' in subject_lower or 'fav' in subject_lower) and query_property in subject_lower:
                    logging.info(f"[DIRECT FAVORITE MATCH] Found perfect match: '{fact.subject}' for query '{user_query}'")
                    enhanced_fact = add_confidence_tracking_to_fact(fact)
                    confidence = getattr(enhanced_fact, 'confidence', 1.0)
                    return f"Your favorite {query_property} is {enhanced_fact.object} (confidence: {confidence:.2f})"
            
            # If no direct match, fall through to general matching
            logging.info(f"[DIRECT FAVORITE MATCH] No direct match found for favorite {query_property}")
        
        print(f"[FACT LOOP] Starting to evaluate {len(triplets)} facts...")
        for fact in triplets:
            # IMPROVED MATCHING: Focus on main object in compound queries
            print(f"[EVALUATING FACT] {fact.subject} | {fact.predicate} | {fact.object}")
            
            # Score based on how well the fact matches the main object being asked about
            object_match_score = 0.0
            if main_object:
                # Check if main object appears in fact's object field (direct substring match)
                if main_object in fact.object.lower():
                    object_match_score = 1.0  # Perfect match
                    logging.info(f"[OBJECT MATCH] Direct: '{main_object}' found in '{fact.object}' → score=1.0")
                # Check if main object is the singular/plural form (e.g., "car" vs "cars")
                elif (main_object + 's') in fact.object.lower() or (main_object[:-1] if main_object.endswith('s') else main_object + 's') in fact.object.lower():
                    object_match_score = 1.0  # Singular/plural match
                    logging.info(f"[OBJECT MATCH] Plural/singular: '{main_object}' matches '{fact.object}' → score=1.0")
                # Check if main object appears as a word boundary (e.g., "car" in "black cars")
                elif f" {main_object}" in f" {fact.object.lower()}" or f"{main_object} " in f"{fact.object.lower()} ":
                    object_match_score = 1.0  # Word boundary match
                    logging.info(f"[OBJECT MATCH] Word boundary: '{main_object}' found as word in '{fact.object}' → score=1.0")
                # Check for common synonyms using fast lookup table
                else:
                    # Fast synonym lookup table for common cases
                    synonym_map = {
                        'feline': ['cat', 'cats', 'kitten', 'kittens'],
                        'canine': ['dog', 'dogs', 'puppy', 'puppies'],
                        'automobile': ['car', 'cars', 'vehicle', 'vehicles'],
                        'vehicle': ['car', 'cars', 'automobile', 'automobiles'],
                        'beverage': ['drink', 'drinks'],
                        'food': ['meal', 'meals', 'dish', 'dishes'],
                        'color': ['colour', 'hue', 'shade'],
                        'movie': ['film', 'films', 'movies'],
                        'book': ['novel', 'novels', 'books']
                    }
                    
                    # Check if main_object has synonyms that appear in fact.object
                    synonym_found = False
                    if main_object in synonym_map:
                        for synonym in synonym_map[main_object]:
                            if synonym in fact.object.lower():
                                object_match_score = 1.0  # Synonym match
                                logging.info(f"[OBJECT MATCH] Synonym: '{main_object}' → '{synonym}' found in '{fact.object}' → score=1.0")
                                synonym_found = True
                                break
                    
                    # Also check reverse mapping (e.g., if fact contains "feline" and we're asking about "cat")
                    if not synonym_found:
                        for synonym_key, synonym_list in synonym_map.items():
                            if main_object in synonym_list and synonym_key in fact.object.lower():
                                object_match_score = 1.0  # Reverse synonym match
                                logging.info(f"[OBJECT MATCH] Reverse synonym: '{main_object}' ← '{synonym_key}' found in '{fact.object}' → score=1.0")
                                synonym_found = True
                                break
                    
                    # Check individual words in fact.object for synonyms
                    if not synonym_found:
                        fact_words = fact.object.lower().split()
                        for word in fact_words:
                            if main_object in synonym_map and word in synonym_map[main_object]:
                                object_match_score = 1.0  # Word-level synonym match
                                logging.info(f"[OBJECT MATCH] Word synonym: '{main_object}' → '{word}' found in '{fact.object}' → score=1.0")
                                synonym_found = True
                                break
                            # Check reverse word-level synonyms
                            for synonym_key, synonym_list in synonym_map.items():
                                if main_object in synonym_list and word == synonym_key:
                                    object_match_score = 1.0  # Reverse word synonym match
                                    logging.info(f"[OBJECT MATCH] Reverse word synonym: '{main_object}' ← '{word}' found in '{fact.object}' → score=1.0")
                                    synonym_found = True
                                    break
                            if synonym_found:
                                break
                    
                    # If no fast synonym found, use semantic similarity as fallback
                    if not synonym_found:
                        object_match_score = calculate_semantic_similarity(main_object, fact.object)
                        logging.info(f"[OBJECT MATCH] Semantic: '{main_object}' vs '{fact.object}' → score={object_match_score:.2f}")
            else:
                logging.info(f"[OBJECT MATCH] No main_object specified")
            
            # Score based on subject field matching  
            subject_match_score = 0.0
            if main_object:
                # Check if main object appears in subject (e.g., "user car preference", "user fav car color")
                if main_object in fact.subject.lower():
                    subject_match_score = 0.8
                    logging.info(f"[SUBJECT MATCH] Direct: '{main_object}' found in '{fact.subject}' → score=0.8")
                # Check for word boundary match in subject
                elif f" {main_object}" in f" {fact.subject.lower()}" or f"{main_object} " in f"{fact.subject.lower()} ":
                    subject_match_score = 0.8
                    logging.info(f"[SUBJECT MATCH] Word boundary: '{main_object}' found as word in '{fact.subject}' → score=0.8")
                else:
                    subject_match_score = calculate_semantic_similarity(main_object or '', fact.subject)
                    logging.info(f"[SUBJECT MATCH] Semantic: '{main_object}' vs '{fact.subject}' → score={subject_match_score:.2f}")
            else:
                logging.info(f"[SUBJECT MATCH] No main_object specified")
            
            # Score based on property matching if specified
            property_match_score = 1.0  # Default high if no specific property
            if query_property:
                # For property queries, check if the property context makes sense
                property_match_score = calculate_semantic_similarity(query_property, fact.subject)
                # Also check if property appears in the fact anywhere
                if query_property in fact.subject.lower() or query_property in fact.predicate.lower():
                    property_match_score = max(property_match_score, 0.7)
            
            # WEIGHTED SCORING: Prioritize object matches for compound queries
            if is_property_query and main_object:
                # For "what color dogs" prioritize facts about dogs
                score = 0.8 * object_match_score + 0.2 * subject_match_score
                logging.info(f"[SCORING] Property query mode: 0.8×{object_match_score:.2f} + 0.2×{subject_match_score:.2f} = {score:.2f}")
            else:
                # Traditional scoring for simple queries  
                score = 0.6 * subject_match_score + 0.4 * object_match_score
                logging.info(f"[SCORING] Simple query mode: 0.6×{subject_match_score:.2f} + 0.4×{object_match_score:.2f} = {score:.2f}")
            
            # SEMANTIC INTENT BOOST: For property queries, heavily favor facts that mention both the object AND property in the subject
            semantic_intent_boost = 0.0
            if is_property_query and main_object and query_property:
                subject_lower = fact.subject.lower()
                # Perfect semantic match: both object and property appear in subject
                if main_object in subject_lower and query_property in subject_lower:
                    semantic_intent_boost = 2.0  # Very strong boost for perfect semantic match
                    logging.info(f"[Semantic Intent] Perfect match: both '{main_object}' and '{query_property}' in '{fact.subject}' (+{semantic_intent_boost})")
                # Partial semantic match: subject is about the object in context of property
                elif query_property in subject_lower and ('fav' in subject_lower or 'favorite' in subject_lower):
                    semantic_intent_boost = 1.5  # Strong boost for preference facts with property
                    logging.info(f"[Semantic Intent] Strong preference match: '{query_property}' + preference in '{fact.subject}' (+{semantic_intent_boost})")
            
            score += semantic_intent_boost
            
            # SPECIFICITY BOOST: Heavily favor facts that mention the specific object
            specificity_boost = 0.0
            if main_object and main_object in fact.object.lower():
                specificity_boost = 1.5  # Very strong boost for exact object matches
                logging.info(f"[Specificity Boost] +{specificity_boost} for '{main_object}' in '{fact.object}'")
            
            score += specificity_boost
            
            # PRIORITY BOOST: For color queries, heavily favor explicit color preference facts
            priority_boost = 0.0
            if is_color_query:
                # Boost explicit color facts like "user fav color" or "user favorite color" 
                if 'color' in fact.subject.lower() and ('fav' in fact.subject.lower() or 'favorite' in fact.subject.lower()):
                    priority_boost = 1.0  # Strong boost for explicit color preferences
                    
                    # Additional recency boost - prefer "fav color" over "favorite color" (more recent/informal)
                    if 'fav color' in fact.subject.lower():
                        priority_boost += 0.2  # Slight additional boost for "fav color" (likely more recent)
                    
                    logging.info(f"[Color Priority] Boosting explicit color fact: '{fact.subject}' with boost {priority_boost}")
                # FIXED: Only penalize general "like" statements that are UNRELATED to the query context
                elif fact.predicate.lower() == 'like' and 'color' not in fact.subject.lower():
                    # Check if this fact is actually relevant to the main object being asked about
                    fact_is_relevant_to_query = False
                    if main_object and main_object in fact.object.lower():
                        fact_is_relevant_to_query = True
                        logging.info(f"[Color Priority] Fact is relevant: '{main_object}' found in '{fact.object}'")
                    
                    # Only penalize if the fact is completely unrelated to the query context
                    if not fact_is_relevant_to_query:
                        priority_boost = -0.5  # Penalty for non-color "like" facts in color queries
                        logging.info(f"[Color Priority] Penalizing unrelated non-color like fact: '{fact.subject} {fact.predicate} {fact.object}'")
                    else:
                        # Don't penalize relevant facts, even if they're general "like" statements
                        logging.info(f"[Color Priority] NOT penalizing relevant fact: '{fact.subject} {fact.predicate} {fact.object}'")
            
            final_score = score + priority_boost
            
            candidates.append((fact, final_score, object_match_score, subject_match_score))
            print(f"[CANDIDATE] '{fact.subject}' '{fact.predicate}' '{fact.object}' → score={final_score:.2f} (obj={object_match_score:.2f}, subj={subject_match_score:.2f}, spec={specificity_boost:.2f}, prio={priority_boost:.2f})")
            if final_score > best_score:
                best_score = final_score
                best_fact = fact
                print(f"[NEW BEST] Updated best fact to: '{fact.subject}' '{fact.predicate}' '{fact.object}' with score {final_score:.2f}")
        # Log all candidates and the final selection
        if best_fact and best_score > 0.4:  # threshold for a good match (lowered for fuzzy matching)
            logging.info(f"[FINAL SELECTION] Chose: '{best_fact.subject}' '{best_fact.predicate}' '{best_fact.object}' (score={best_score:.2f})")
            logging.info(f"[ALL CANDIDATES] Total candidates evaluated: {len(candidates)}")
            for i, (fact, score, obj_score, subj_score) in enumerate(sorted(candidates, key=lambda x: x[1], reverse=True)[:3]):
                logging.info(f"  {i+1}. '{fact.subject}' '{fact.predicate}' '{fact.object}' → {score:.2f}")
            
            # Add confidence tracking to the selected fact
            enhanced_fact = add_confidence_tracking_to_fact(best_fact)
            
            # Use confidence-aware response formatting
            confidence = getattr(enhanced_fact, 'confidence', 1.0)
            
            # PRESERVE SUBJECT CONTEXT: Include subject information for clarity
            subject_context = ""
            if enhanced_fact.subject.lower() != "user":
                # Extract the meaningful part of the subject (e.g., "fav cat color" from "user fav cat color")
                subject_parts = enhanced_fact.subject.lower().split()
                if "user" in subject_parts:
                    # Remove "user" and join the rest
                    meaningful_parts = [part for part in subject_parts if part != "user"]
                    if meaningful_parts:
                        subject_context = f"user's {' '.join(meaningful_parts)} "
                else:
                    subject_context = f"{enhanced_fact.subject} "
            
            # Format response with confidence level
            if subject_context:
                base_response = f"{subject_context}{enhanced_fact.predicate} {enhanced_fact.object}"
            else:
                base_response = f"user {enhanced_fact.predicate} {enhanced_fact.object}"
            
            return f"{base_response} (confidence: {confidence:.2f})"
        else:
            logging.info(f"[NO MATCH] Best score {best_score:.2f} below threshold 0.4. Candidates: {len(candidates)}")
            # Use uncertainty response instead of generic fallback
            return generate_uncertainty_response(user_query, "no_memory")
    # Fallback to original context builder
    print(f"[FALLBACK] Using old_build_memory_context for non-preference query: '{user_query}'")
    return old_build_memory_context(triplets, max_tokens)

# Patch get_most_recent_resolved_fact to use both subject and predicate for matching
def get_most_recent_resolved_fact(facts, subject, predicate=None):
    from storage.memory_utils import _normalize_subject
    subject = _normalize_subject(subject)
    filtered = [
        f for f in facts
        if _normalize_subject(f.subject) == subject
        and (predicate is None or predicate in f.predicate.lower())
        and (getattr(f, 'contradiction_score', 0.0) < 0.3)
    ]
    if not filtered:
        return None
    # Sort by confidence, then timestamp
    filtered.sort(key=lambda f: (-(getattr(f, 'decayed_confidence', f.confidence)), f.timestamp), reverse=True)
    return filtered[0]
# --- END PATCH ---

def validate_memory_response(user_query: str, memory_context: str, candidate_response: str, confidence_threshold: float = 0.6) -> dict:
    """
    Validate that a response is based on actual memory facts, not fabricated.
    
    Args:
        user_query: The user's question
        memory_context: Available memory facts  
        candidate_response: The proposed response
        confidence_threshold: Minimum confidence required (0.0-1.0)
        
    Returns:
        Dict with validation results and recommended response
    """
    validation_result = {
        "is_valid": False,
        "confidence": 0.0,
        "has_memory": False,
        "recommended_response": "",
        "uncertainty_level": "high",
        "issues": []
    }
    
    # Check if there's actual memory context
    if not memory_context or memory_context.strip() == "":
        validation_result["has_memory"] = False
        validation_result["issues"].append("no_memory_available")
        validation_result["recommended_response"] = generate_uncertainty_response(user_query, "no_memory")
        return validation_result
    
    # Check for common fabrication patterns
    fabrication_indicators = [
        "based on", "you seem to", "you appear to", "it seems like",
        "you might", "you probably", "i think you", "you typically"
    ]
    
    response_lower = candidate_response.lower()
    for indicator in fabrication_indicators:
        if indicator in response_lower and "confidence:" not in response_lower:
            validation_result["issues"].append(f"fabrication_indicator: {indicator}")
    
    # Check if response contains memory confidence markers
    if "confidence:" in response_lower:
        try:
            import re
            conf_match = re.search(r'confidence:\s*([0-9.]+)', response_lower)
            if conf_match:
                confidence = float(conf_match.group(1))
                validation_result["confidence"] = confidence
                validation_result["has_memory"] = True
                
                if confidence >= confidence_threshold:
                    validation_result["is_valid"] = True
                    validation_result["uncertainty_level"] = "low" if confidence > 0.8 else "medium"
                else:
                    validation_result["issues"].append(f"low_confidence: {confidence}")
                    validation_result["recommended_response"] = generate_uncertainty_response(user_query, "low_confidence", confidence)
        except:
            validation_result["issues"].append("invalid_confidence_format")
    
    # IMPROVED: Much broader memory indicators and content matching
    memory_indicators = [
        "you like", "your favorite", "you prefer", "you said", "you told me", 
        "i remember", "according to", "user like", "user favorite",
        "last time", "previously", "earlier", "before", "you have expressed",
        "based on", "i know that", "with a confidence", "confidence of",
        "you've been", "you are", "from our conversations"
    ]
    
    has_memory_content = any(indicator in response_lower for indicator in memory_indicators)
    
    # ENHANCED: Check if response references actual memory content - more sophisticated matching
    if memory_context:
        # Extract key objects/entities from memory context
        memory_words = memory_context.lower().split()
        significant_words = [word for word in memory_words if len(word) > 3 and word not in ['user', 'like', 'confidence', 'with']]
        
        # Check if any significant memory content appears in the response
        memory_content_referenced = any(word in response_lower for word in significant_words)
        
        if memory_content_referenced:
            has_memory_content = True
            validation_result["has_memory"] = True
            logging.info(f"[MEMORY VALIDATION] Found memory content reference: memory words {significant_words} in response")
        
    # CRITICAL FIX: Allow responses that reference memory content even without explicit confidence
    if has_memory_content and not validation_result["has_memory"]:
        validation_result["has_memory"] = True
        validation_result["confidence"] = 0.7  # Assume reasonable confidence for memory responses
        validation_result["is_valid"] = True
        validation_result["uncertainty_level"] = "medium"
        logging.info(f"[MEMORY VALIDATION] Accepting response with memory content indicators")
    
    # Only flag as fabrication if no memory content AND no memory context available
    if not validation_result["has_memory"] and not has_memory_content:
        validation_result["issues"].append("no_memory_markers")
        validation_result["recommended_response"] = generate_uncertainty_response(user_query, "uncertain")
    
    return validation_result


def generate_uncertainty_response(user_query: str, reason: str, confidence: float = 0.0) -> str:
    """
    Generate appropriate uncertainty responses instead of fabricating facts.
    
    Args:
        user_query: The user's question
        reason: Why we're uncertain ("no_memory", "low_confidence", "uncertain")
        confidence: Confidence level if available
        
    Returns:
        Appropriate uncertainty response
    """
    # Extract what the user is asking about
    query_lower = user_query.lower()
    
    # Determine what they're asking about for context
    asking_about = ""
    for pattern in QUERY_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            asking_about = match.group(1)
            break
    
    if reason == "no_memory":
        return f"I don't have any information about {asking_about} in my memory. Could you tell me what it is?"
    
    elif reason == "low_confidence":
        return f"I have some information about {asking_about}, but I'm not confident enough (confidence: {confidence:.2f}) to give you a definitive answer. Could you remind me?"
    
    elif reason == "uncertain":
        return f"I'm not certain about {asking_about}. Rather than guess, could you tell me what it is?"
    
    else:
        return f"I don't have reliable information about {asking_about}. What would you like me to remember?"


def enhance_system_prompt_with_validation(base_prompt: str) -> str:
    """
    Enhance the system prompt to prevent fabrication and encourage uncertainty handling.
    
    Args:
        base_prompt: The existing system prompt
        
    Returns:
        Enhanced prompt with validation instructions
    """
    validation_instructions = """

CRITICAL MEMORY VALIDATION RULES:
- NEVER fabricate or guess facts about the user
- If you don't have memory evidence, say "I don't know" or "I don't have that information"
- Only state facts that come with confidence scores from memory
- When uncertain, ask the user to clarify rather than guessing
- Always include confidence levels when stating remembered facts
- Use phrases like "I don't have information about..." instead of "you seem to" or "you might"

UNCERTAINTY INDICATORS TO USE:
- "I don't have information about..."
- "I'm not certain about..."
- "Could you tell me about..."
- "I don't recall..."

NEVER USE THESE FABRICATION PHRASES:
- "You seem to..."
- "You appear to..."
- "You might..."
- "You probably..."
- "Based on your..."
- "It seems like..."

When in doubt, always choose uncertainty over fabrication."""

    return base_prompt + validation_instructions


def add_confidence_tracking_to_fact(fact: TripletFact) -> TripletFact:
    """
    Add confidence tracking metadata to a fact.
    
    Args:
        fact: The TripletFact to enhance
        
    Returns:
        Enhanced fact with confidence tracking
    """
    # Calculate confidence based on frequency, recency, and contradiction score
    base_confidence = getattr(fact, 'confidence', 1.0) or 0.0  # Handle None values
    frequency = getattr(fact, 'frequency', 1) or 1  # Handle None values
    frequency_boost = min(0.2, frequency * 0.05) if frequency and frequency > 0 else 0  # Up to 0.2 boost
    
    # Recency factor (facts decay over time)
    try:
        from datetime import datetime
        if hasattr(fact, 'timestamp') and fact.timestamp:
            fact_time = datetime.fromisoformat(fact.timestamp.replace("Z", "+00:00"))
            days_old = (datetime.now().replace(tzinfo=fact_time.tzinfo) - fact_time).days
            recency_factor = max(0.5, 1.0 - (days_old * 0.01))  # Slow decay
        else:
            recency_factor = 1.0
    except:
        recency_factor = 1.0
    
    # Contradiction penalty
    contradiction_penalty = getattr(fact, 'contradiction_score', 0.0) * 0.5
    
    # Calculate final confidence
    final_confidence = (base_confidence + frequency_boost) * recency_factor - contradiction_penalty
    final_confidence = max(0.0, min(1.0, final_confidence))
    
    # Add confidence metadata
    fact.confidence = final_confidence
    fact.uncertainty_level = (
        "low" if final_confidence > 0.8 else
        "medium" if final_confidence > 0.6 else
        "high"
    )
    
    return fact


def format_response_with_confidence(fact: TripletFact, context: str = "") -> str:
    """
    Format a response with appropriate confidence indicators.
    
    Args:
        fact: The fact to present
        context: Additional context
        
    Returns:
        Formatted response with confidence level
    """
    confidence = getattr(fact, 'confidence', 1.0)
    uncertainty_level = getattr(fact, 'uncertainty_level', 'medium')
    
    # Base response
    if context:
        response = f"{context}: {fact.object}"
    else:
        response = f"{fact.subject} {fact.predicate} {fact.object}"
    
    # Add confidence indicator
    if confidence >= 0.9:
        confidence_text = "I'm very confident"
    elif confidence >= 0.7:
        confidence_text = "I'm fairly confident"
    elif confidence >= 0.5:
        confidence_text = "I think"
    else:
        confidence_text = "I'm not very confident, but I believe"
    
    return f"{confidence_text} that {response.lower()} (confidence: {confidence:.2f})"
