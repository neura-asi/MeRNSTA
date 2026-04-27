#!/usr/bin/env python3
"""
Centralized configuration for MeRNSTA memory system
Loads all settings from config.yaml - NO HARDCODING
"""

import hashlib
import socket
import os
import yaml
import logging
from typing import Dict, List, Any, Optional

# Load configuration from config.yaml
def _load_config() -> Dict[str, Any]:
    """Load all configuration from config.yaml"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config.yaml: {e}")
        raise RuntimeError(f"Cannot load configuration: {e}")

# Load the main configuration
_cfg = _load_config()

# === EXTRACT ALL CONFIGURATION VALUES FROM YAML ===

# Volatility thresholds for fact stability classification
VOLATILITY_THRESHOLDS = _cfg.get("volatility_thresholds", {})

# Default thresholds for method parameters
DEFAULT_THRESHOLDS = _cfg.get("default_thresholds", {})

# Prompt formatting templates
PROMPT_FORMAT = _cfg.get("prompt_format", {})

# Entity to category mappings
DEFAULT_ENTITY_CATEGORIES = _cfg.get("entity_categories", {})

# Category display order (most important first)
CATEGORY_ORDER = _cfg.get("category_order", [])

# Confidence thresholds for visual indicators
CONFIDENCE_THRESHOLDS = _cfg.get("confidence_thresholds", {})

# Volatility visual indicators
VOLATILITY_ICONS = _cfg.get("volatility_icons", {})

# Confidence visual indicators
CONFIDENCE_ICONS = _cfg.get("confidence_icons", {})

# Database configuration
DATABASE_CONFIG = _cfg.get("database", {})

# Memory context configuration
MEMORY_CONFIG = _cfg.get("memory_config", {})

# Memory routing modes configuration
MEMORY_ROUTING_MODES = _cfg.get("memory_routing_modes", {})

# Default memory routing mode
DEFAULT_MEMORY_MODE = _cfg.get("default_memory_mode", "MAC")

# Personality-based memory biasing configuration
PERSONALITY_PROFILES = _cfg.get("personality_profiles", {})

# Default personality
DEFAULT_PERSONALITY = _cfg.get("default_personality", "neutral")

# Behavior settings
_behavior = _cfg.get("behavior", {})
AUTO_RECONCILE = _behavior.get("auto_reconcile", True)
EMOTION_BIAS = _behavior.get("emotion_bias", True)
enable_compression = _behavior.get("enable_compression", True)
semantic_drift_threshold = _behavior.get("semantic_drift_threshold", 0.35)

# Default values for various operations
DEFAULT_VALUES = _cfg.get("default_values", {})

# Causal linkage threshold and parameters
# Restored to reasonable threshold after confirming weighted formula works
CAUSAL_LINK_THRESHOLD = DEFAULT_VALUES.get("causal_link_threshold", 0.25)
TEMPORAL_DECAY_LAMBDA = DEFAULT_VALUES.get("temporal_decay_lambda", 0.1)

# Contradiction detection settings
_contradiction = _cfg.get("contradiction_detection", {})
CONTRADICTION_SCORE_THRESHOLD = _contradiction.get("score_threshold", 0.85)
CONTRADICTION_DETECTION_STRICT = _contradiction.get("detection_strict", True)

# Enhanced contradiction detection thresholds
_contradiction_thresholds = _cfg.get("contradiction_thresholds", {})
SEMANTIC_SIMILARITY_THRESHOLD = _contradiction_thresholds.get("semantic_similarity", 0.7)
PREFERENCE_CONFLICT_CONFIDENCE = _contradiction_thresholds.get("preference_conflict", 0.8)
DIRECT_CONTRADICTION_CONFIDENCE = _contradiction_thresholds.get("direct_contradiction", 0.9)
VOLATILITY_DETECTION_THRESHOLD = _contradiction_thresholds.get("volatility_threshold", 0.6)
MAX_WORDS_SIMPLE_OBJECT = _contradiction_thresholds.get("max_words_simple_object", 2)

# Preference categories for conflict detection
PREFERENCE_CATEGORIES = _cfg.get("preference_categories", {})
BEVERAGES = PREFERENCE_CATEGORIES.get("beverages", ["tea", "coffee"])
FOODS = PREFERENCE_CATEGORIES.get("foods", ["pizza", "pasta"])
COLORS = PREFERENCE_CATEGORIES.get("colors", ["red", "blue"])
ACTIVITIES = PREFERENCE_CATEGORIES.get("activities", ["reading", "writing"])
SPORTS = PREFERENCE_CATEGORIES.get("sports", ["football", "basketball"])
MUSIC_GENRES = PREFERENCE_CATEGORIES.get("music_genres", ["rock", "pop"])

# Reflective prompting settings
_reflective = _cfg.get("reflective_prompting", {})
REFLECTIVE_PROMPTING = _reflective.get("enabled", True)
REFLECTION_VOLATILITY_THRESHOLD = _reflective.get("volatility_threshold", 0.5)
REFLECTION_SLOPE_THRESHOLD = _reflective.get("slope_threshold", 0.5)
REFLECTIVE_FORECASTING = _reflective.get("forecasting", True)

# Dynamic personality settings
PERSONALITY_MODE = _cfg.get("personality_mode", "auto")

# Dynamic personality clustering parameters
PERSONALITY_CLUSTERING = _cfg.get("personality_clustering", {})

# Adaptive threshold adjustment based on data quality
PERSONALITY_ADAPTATION = _cfg.get("personality_adaptation", {})

# Fact extraction patterns with confidence scoring
def _convert_extraction_patterns(patterns_config: List[Dict]) -> List[tuple]:
    """Convert YAML pattern config to regex tuples"""
    return [(p["pattern"], p["confidence"]) for p in patterns_config]

FACT_EXTRACTION_PATTERNS = _convert_extraction_patterns(
    _cfg.get("fact_extraction_patterns", [])
)

# Fallback patterns with lower confidence
FALLBACK_EXTRACTION_PATTERNS = _convert_extraction_patterns(
    _cfg.get("fallback_extraction_patterns", [])
)

# Pattern-based categorization keywords
CATEGORY_PATTERNS = _cfg.get("category_patterns", {})

# System settings
_system = _cfg.get("system", {})
VERBOSITY_LEVEL = _system.get("verbosity_level", "normal")

# Question words to skip during subject extraction
QUESTION_WORDS = _cfg.get("question_words", [])

# Similarity threshold for semantic search
SIMILARITY_THRESHOLD = _cfg.get("similarity_threshold", 0.75)

# Database connection pool settings
_database = _cfg.get("database", {})
MAX_CONNECTIONS = _database.get("max_connections", 10)
RETRY_DELAY = _database.get("retry_delay", 0.1)
RETRY_ATTEMPTS = _database.get("retry_attempts", 5)

# Multi-Modal Memory Config
_multimodal = _cfg.get("multimodal", {})
CLIP_MODEL = _multimodal.get("clip_model", "openai/clip-vit-base-patch32")
WHISPER_MODEL = _multimodal.get("whisper_model", "openai/whisper-base")

# Network configuration
_network = _cfg.get("network", {})
ollama_host = _network.get("ollama_host", "http://127.0.0.1:11434")
embedding_model = _multimodal.get("embedding_model", "mistral")
MEDIA_STORAGE_PATH = _multimodal.get("media_storage_path", "media/")
MULTIMODAL_SIMILARITY_THRESHOLD = _multimodal.get("similarity_threshold", 0.7)

# Profile and session settings
PROFILE_ID_SOURCE = _cfg.get("profile_id_source", "ip_hash")
CROSS_SESSION_SEARCH_ENABLED = _cfg.get("cross_session_search_enabled", True)

# Code analysis
code_markers = _cfg.get("code_markers", ["def ", "class "])
question_words = _cfg.get("question_words", [])
similarity_threshold = _cfg.get("similarity_threshold", 0.7)
max_cluster_size = _multimodal.get("max_cluster_size", 10)

# API Server Port Config
api_port = _network.get("api_port", 8000)
port_retry_attempts = _network.get("port_retry_attempts", 5)

# Dashboard and code evolution
_dashboard = _cfg.get("dashboard", {})
dashboard_port = _network.get("dashboard_port", 8001)
pagination_limit = _dashboard.get("pagination_limit", 50)

_code_evolution = _cfg.get("code_evolution", {})
require_confirmation = _code_evolution.get("require_confirmation", True)
max_patch_size = _code_evolution.get("max_patch_size", 1000)

QUERY_PATTERNS = _cfg.get("query_patterns", [])
IMPERATIVE_VERBS = _cfg.get("imperative_verbs", [])
IMPERATIVE_PATTERNS = _cfg.get("imperative_patterns", {})
CONVERSATIONAL_PATTERNS = _cfg.get("conversational_patterns", [])
STATEMENT_INDICATORS = _cfg.get("statement_indicators", [])

def get_user_profile_id():
    """Get user profile ID based on configuration"""
    if PROFILE_ID_SOURCE == "ip_hash":
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
        return hashlib.sha256(ip.encode()).hexdigest()[:12]
    elif PROFILE_ID_SOURCE == "env":
        return os.environ.get("MERNSTA_PROFILE_ID", "default_profile")
    elif callable(PROFILE_ID_SOURCE):
        return PROFILE_ID_SOURCE()
    else:
        return "default_profile"

# === CONFIGURATION VALIDATION ===
def validate_config():
    """Validate that all required configuration values are present"""
    required_sections = [
        "volatility_thresholds", "default_thresholds", "prompt_format",
        "entity_categories", "category_order", "confidence_thresholds",
        "volatility_icons", "confidence_icons", "database", "memory_config",
        "memory_routing_modes", "personality_profiles", "default_values",
        "fact_extraction_patterns", "fallback_extraction_patterns",
        "category_patterns", "question_words", "multimodal", "network"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in _cfg or not _cfg[section]:
            missing_sections.append(section)
    
    if missing_sections:
        raise RuntimeError(f"Missing required configuration sections: {missing_sections}")
    
    logging.info("✅ Configuration validation passed - all sections present")

# Validate configuration on import
validate_config()

# Log successful configuration load
# === PUBLIC API ===

def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return _cfg

def reload_config() -> Dict[str, Any]:
    """Reload configuration from config.yaml and return the new config."""
    global _cfg
    _cfg = _load_config()
    return _cfg

logging.info(f"✅ Configuration loaded from config.yaml - {len(_cfg)} sections")
logging.info(f"📊 Loaded {len(FACT_EXTRACTION_PATTERNS)} extraction patterns")
logging.info(f"👤 Loaded {len(PERSONALITY_PROFILES)} personality profiles")
logging.info(f"🎯 Default memory mode: {DEFAULT_MEMORY_MODE}")
logging.info(f"🔧 Ollama host: {ollama_host}")

# === Derived getters ===
def get_token_budget(default: int = 512) -> int:
    """Return the token budget for LLM prompts.

    Reads `llm.max_tokens` if present; otherwise falls back to
    `memory_config.max_tokens`; final fallback is the provided default.
    """
    try:
        llm_cfg = _cfg.get("llm", {}) or {}
        if isinstance(llm_cfg.get("max_tokens"), int) and llm_cfg.get("max_tokens") > 0:
            return int(llm_cfg["max_tokens"])
        mem_cfg = _cfg.get("memory_config", {}) or {}
        if isinstance(mem_cfg.get("max_tokens"), int) and mem_cfg.get("max_tokens") > 0:
            return int(mem_cfg["max_tokens"])
    except Exception:
        pass
    return int(default)
