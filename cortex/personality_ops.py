"""
Personality operations for MeRNSTA cortex package.
Handles personality-based confidence adjustments and behavioral modifications.
"""

import logging
import yaml
from typing import Dict, List, Any
from config.environment import get_settings

def load_personality_profiles() -> Dict[str, Any]:
    """Load personality profiles from config.yaml."""
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config.get('personality_profiles', {})
    except Exception as e:
        logging.warning(f"Failed to load personality profiles from config.yaml: {e}")
        # Return default personalities
        return {
            "neutral": {
                "multiplier": 1.0,
                "description": "Balanced and objective approach",
                "confidence_adjustment": 0.0
            },
            "skeptical": {
                "multiplier": 1.5,
                "description": "Questions facts and reduces confidence",
                "confidence_adjustment": -0.05
            },
            "enthusiastic": {
                "multiplier": 0.8,
                "description": "Optimistic and encouraging",
                "confidence_adjustment": 0.05
            },
            "analytical": {
                "multiplier": 1.2,
                "description": "Detail-oriented and methodical",
                "confidence_adjustment": 0.02
            },
            "empathetic": {
                "multiplier": 0.9,
                "description": "Understanding and compassionate",
                "confidence_adjustment": 0.01
            },
            "cautious": {
                "multiplier": 1.3,
                "description": "Careful and conservative",
                "confidence_adjustment": -0.03
            }
        }

def apply_personality(facts: List[Dict[str, Any]], personality: str) -> List[Dict[str, Any]]:
    """
    Apply personality-based confidence adjustments to facts.
    
    Args:
        facts: List of fact dictionaries with confidence scores
        personality: Personality profile name
        
    Returns:
        Modified facts with adjusted confidence scores
    """
    if not facts or not personality:
        return facts
    
    personality_profiles = load_personality_profiles()
    
    if personality not in personality_profiles:
        logging.warning(f"Unknown personality '{personality}', using neutral")
        personality = "neutral"
    
    profile = personality_profiles[personality]
    multiplier = profile.get("multiplier", 1.0)
    
    adjusted_facts = []
    
    for fact in facts:
        if isinstance(fact, dict) and "confidence" in fact:
            original_confidence = fact["confidence"]
            
            # Apply multiplier
            adjusted_confidence = original_confidence / multiplier
            
            # Ensure confidence stays within valid bounds [0.0, 1.0]
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
            
            # Create new fact with adjusted confidence
            adjusted_fact = fact.copy()
            adjusted_fact["confidence"] = adjusted_confidence
            adjusted_fact["original_confidence"] = original_confidence
            adjusted_fact["personality_applied"] = personality
            
            adjusted_facts.append(adjusted_fact)
            
            logging.debug(f"Personality '{personality}': {original_confidence:.3f} -> {adjusted_confidence:.3f}")
        else:
            # Pass through facts without confidence scores unchanged
            adjusted_facts.append(fact)
    
    return adjusted_facts

def get_personality_description(personality: str) -> str:
    """Get description for a personality profile."""
    personality_profiles = load_personality_profiles()
    
    if personality in personality_profiles:
        return personality_profiles[personality].get("description", "Unknown personality")
    else:
        return "Unknown personality"

def get_available_personalities() -> List[str]:
    """Get list of available personality profile names."""
    personality_profiles = load_personality_profiles()
    return list(personality_profiles.keys())

def validate_personality(personality: str) -> bool:
    """Check if a personality profile exists."""
    personality_profiles = load_personality_profiles()
    return personality in personality_profiles

def get_personality_multiplier(personality: str) -> float:
    """Get the confidence multiplier for a personality."""
    personality_profiles = load_personality_profiles()
    
    if personality in personality_profiles:
        return personality_profiles[personality].get("multiplier", 1.0)
    else:
        return 1.0

def get_personality_adjustment(personality: str) -> float:
    """Get the confidence adjustment for a personality."""
    personality_profiles = load_personality_profiles()
    
    if personality in personality_profiles:
        return personality_profiles[personality].get("confidence_adjustment", 0.0)
    else:
        return 0.0

def apply_personality_to_response(response_text: str, personality: str, confidence: float) -> str:
    """
    Modify response text based on personality and confidence.
    
    Args:
        response_text: Original response text
        personality: Personality profile name
        confidence: Confidence score after personality adjustment
        
    Returns:
        Modified response text with personality-appropriate language
    """
    if not personality or personality == "neutral":
        return response_text
    
    personality_profiles = load_personality_profiles()
    
    if personality not in personality_profiles:
        return response_text
    
    # Add personality-specific language based on confidence level
    if personality == "skeptical":
        if confidence < 0.5:
            return f"I'm quite uncertain about this, but {response_text.lower()}"
        elif confidence < 0.7:
            return f"I have some doubts, but {response_text.lower()}"
        else:
            return f"While I'm somewhat confident, {response_text.lower()}"
    
    elif personality == "enthusiastic":
        if confidence > 0.8:
            return f"I'm excited to tell you that {response_text.lower()}!"
        elif confidence > 0.6:
            return f"Great news! {response_text}"
        else:
            return f"I think {response_text.lower()}"
    
    elif personality == "cautious":
        if confidence < 0.6:
            return f"I should mention that I'm not very certain, but {response_text.lower()}"
        elif confidence < 0.8:
            return f"Please note that {response_text.lower()}"
        else:
            return f"Based on available information, {response_text.lower()}"
    
    elif personality == "analytical":
        return f"Based on my analysis, {response_text.lower()}"
    
    elif personality == "empathetic":
        return f"I understand you're asking about this, and {response_text.lower()}"
    
    return response_text

def determine_personality_from_contradiction_level(contradiction_level: float) -> str:
    """
    Suggest a personality based on contradiction level in memory.
    
    Args:
        contradiction_level: Ratio of contradictory facts (0.0 to 1.0)
        
    Returns:
        Suggested personality profile name
    """
    if contradiction_level > 0.3:
        return "skeptical"  # High contradictions warrant skepticism
    elif contradiction_level > 0.15:
        return "cautious"   # Moderate contradictions suggest caution
    elif contradiction_level < 0.05:
        return "analytical" # Low contradictions allow analytical approach
    else:
        return "neutral"    # Default for moderate levels

def get_personality_stats(personality: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a personality profile.
    
    Args:
        personality: Personality profile name
        
    Returns:
        Dictionary with personality statistics and info
    """
    personality_profiles = load_personality_profiles()
    
    if personality not in personality_profiles:
        return {"error": f"Unknown personality: {personality}"}
    
    profile = personality_profiles[personality]
    
    return {
        "name": personality,
        "description": profile.get("description", ""),
        "multiplier": profile.get("multiplier", 1.0),
        "confidence_adjustment": profile.get("confidence_adjustment", 0.0),
        "effect": "increases confidence" if profile.get("multiplier", 1.0) < 1.0 else "decreases confidence",
        "skepticism_level": "high" if profile.get("multiplier", 1.0) > 1.2 else "low" if profile.get("multiplier", 1.0) < 0.9 else "moderate"
    } 