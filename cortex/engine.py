# cortex/engine.py
import numpy as np
import yaml
from typing import List, Tuple, Dict
import math

class CortexEngine:
    """
    Bayesian ranking engine for MeRNSTA memory management.
    Implements the mathematical core from the README.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = yaml.safe_load(open(config_path))
        self.alpha = self.cfg.get("alpha", 0.9)  # Bayesian update rate
        self.gamma = self.cfg.get("gamma", 0.15)  # PPO-tuned contradiction weight
        
    def bayesian_surprise(self, token: str, context: List[str], 
                         current_rank: float) -> float:
        """
        Bayesian Surprise: r_{t+1}(w) = α r_t(w) + (1-α) KL(P(w|C_t) || P(w))
        """
        # Simplified KL divergence calculation
        context_prob = self._estimate_context_probability(token, context)
        prior_prob = self._estimate_prior_probability(token)
        
        if prior_prob > 0 and context_prob > 0:
            kl_div = context_prob * math.log(context_prob / prior_prob)
        else:
            kl_div = 0.0
            
        new_rank = self.alpha * current_rank + (1 - self.alpha) * kl_div
        return max(0.0, new_rank)  # Ensure non-negative
        
    def _estimate_context_probability(self, token: str, context: List[str]) -> float:
        """Estimate P(token | context) using simple frequency"""
        if not context:
            return 0.1  # Default low probability
        context_text = " ".join(context).lower()
        token_lower = token.lower()
        try:
            return context_text.count(token_lower) / len(context_text.split()) if context_text else 0.1
        except ZeroDivisionError:
            return 0.1
        
    def _estimate_prior_probability(self, token: str) -> float:
        """Estimate P(token) using simple frequency"""
        # In a real implementation, this would use a large corpus
        return 0.01  # Default prior probability
        
    def update_rank(self, token: str, context: List[str], 
                   current_rank: float) -> float:
        """Update token rank using Bayesian surprise"""
        return self.bayesian_surprise(token, context, current_rank)
        
    def decay_ranks(self, ranks: Dict[str, float], decay_rate: float = 0.99) -> Dict[str, float]:
        """Apply exponential decay to all ranks"""
        return {token: rank * decay_rate for token, rank in ranks.items()}
        
    def prune_low_rank(self, ranks: Dict[str, float], 
                      threshold: float) -> Dict[str, float]:
        """Remove tokens below rank threshold"""
        return {token: rank for token, rank in ranks.items() 
                if rank >= threshold} 