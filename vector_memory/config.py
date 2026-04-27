# vector_memory/config.py

"""
Configuration management for vector memory backends.
"""

import yaml
import logging
import os
from typing import Optional, Callable

def load_memory_config() -> dict:
    """
    Load memory configuration from config.yaml
    
    Returns:
        Memory configuration dictionary
    """
    try:
        config_path = "configs/config.yaml"
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get memory configuration with defaults
        memory_config = config.get('memory', {})
        return {
            'vector_backend': memory_config.get('vector_backend', 'default'),
            'hybrid_mode': memory_config.get('hybrid_mode', False),
            'fallback_backend': memory_config.get('fallback_backend', 'default')
        }
    except Exception as e:
        logging.warning(f"Failed to load memory config: {e}, using defaults")
        return {
            'vector_backend': 'default',
            'hybrid_mode': False,
            'fallback_backend': 'default'
        }

def get_configured_vectorizer() -> Callable[[str], list]:
    """
    Get the configured vectorizer function based on config.yaml settings.
    
    Returns:
        Vectorizer function that takes text and returns list of floats
    """
    from . import get_vectorizer
    
    config = load_memory_config()
    backend = config['vector_backend']
    
    try:
        vectorizer = get_vectorizer(backend)
        logging.info(f"✅ Using {backend} vector backend")
        return vectorizer
    except Exception as e:
        # Fallback to default if primary backend fails
        fallback = config['fallback_backend']
        logging.warning(f"❌ {backend} backend failed: {e}, falling back to {fallback}")
        try:
            return get_vectorizer(fallback)
        except Exception as fallback_error:
            logging.error(f"❌ Fallback {fallback} also failed: {fallback_error}")
            # Ultimate fallback - use basic embedder
            from scripts.embedder import embed
            import numpy as np
            # Ensure we don't import sentence_transformers implicitly anywhere here
            import sys as _sys
            _sys.modules.pop('sentence_transformers', None)
            def emergency_vectorizer(text: str) -> list:
                embedding = embed(text)
                if isinstance(embedding, np.ndarray):
                    return embedding.tolist()
                return embedding if isinstance(embedding, list) else [0.0] * 384
            return emergency_vectorizer

def get_vectorizer_info() -> dict:
    """
    Get information about the current vectorizer configuration.
    
    Returns:
        Dictionary with vectorizer status and capabilities
    """
    config = load_memory_config()
    
    # Try to get backend info
    try:
        from . import get_vectorizer
        from .hrrformer_adapter import get_hrr_info
        from .vecsymr_adapter import get_vecsymr_info
        from .hlb_adapter import get_hlb_info
        
        backend = config['vector_backend']
        
        if backend == 'hrrformer':
            backend_info = get_hrr_info()
        elif backend == 'vecsymr':
            backend_info = get_vecsymr_info()
        elif backend == 'hlb':
            backend_info = get_hlb_info()
        else:
            backend_info = {
                "backend": "Default (Ollama)",
                "description": "Standard semantic embedding via Ollama API",
                "vector_size": 384,
                "status": "active"
            }
        
        return {
            "current_backend": backend,
            "config": config,
            "backend_info": backend_info
        }
    except Exception as e:
        return {
            "current_backend": "unknown",
            "config": config,
            "error": str(e)
        }