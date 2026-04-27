# cortex/ppo_tuner.py
"""
PPO-Inspired Parameter Tuning for MeRNSTA

This module implements a simplified PPO-inspired approach for tuning the gamma
parameter in contradiction detection. While it uses policy gradient principles
similar to PPO, it is not a full reinforcement learning implementation.

The approach focuses on adaptive parameter adjustment based on contradiction
detection performance feedback, using reward signals to guide gamma updates.
"""
import numpy as np
from typing import List, Dict, Tuple
import yaml
import json
import time
from dataclasses import dataclass
from .contradiction import ContradictionDetector

@dataclass
class ContradictionEvent:
    """Record of a contradiction detection event"""
    timestamp: float
    gamma: float
    token: str
    memory_token: str
    similarity_score: float
    contradiction_score: float
    was_correct: bool  # True if contradiction was actually present
    false_positive: bool  # True if we vetoed a non-contradictory token

class PPOTuner:
    """
    PPO-inspired tuner for gamma parameter based on contradiction detection performance.
    
    Uses simplified policy gradient principles to adapt the gamma parameter.
    This is not a full PPO implementation but rather draws inspiration from
    PPO's reward-based parameter updates.
    
    Reward = contradiction caught, Penalty = false positives.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml", 
                 learning_rate: float = 0.01,
                 gamma_range: Tuple[float, float] = (0.05, 0.5)):
        self.cfg = yaml.safe_load(open(config_path))
        self.learning_rate = learning_rate
        self.gamma_range = gamma_range
        self.current_gamma = self.cfg.get("gamma", 0.15)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Experience buffer
        self.events: List[ContradictionEvent] = []
        self.max_buffer_size = 1000
        
        # Performance tracking
        self.total_contradictions = 0
        self.correct_detections = 0
        self.false_positives = 0
        
    def calculate_reward(self, event: ContradictionEvent) -> float:
        """
        Calculate reward based on contradiction detection performance.
        
        Reward structure:
        - +1.0 for correctly catching a contradiction
        - -2.0 for false positive (vetoing non-contradictory token)
        - +0.1 for correctly allowing non-contradictory token
        - -0.1 for missing a contradiction
        """
        if event.was_correct and not event.false_positive:
            return 1.0  # Correctly caught contradiction
        elif event.false_positive:
            return -2.0  # False positive penalty
        elif not event.was_correct and not event.false_positive:
            return 0.1  # Correctly allowed token
        else:
            return -0.1  # Missed contradiction
            
    def update_gamma(self, event: ContradictionEvent):
        """
        Update gamma using PPO-inspired policy gradient.
        """
        reward = self.calculate_reward(event)
        
        # Simple policy gradient update
        # If reward is positive, increase gamma (more sensitive)
        # If reward is negative, decrease gamma (less sensitive)
        gamma_delta = self.learning_rate * reward
        
        # Clip gamma to valid range
        new_gamma = np.clip(
            self.current_gamma + gamma_delta,
            self.gamma_range[0],
            self.gamma_range[1]
        )
        
        # Update gamma
        old_gamma = self.current_gamma
        self.current_gamma = new_gamma
        
        # Update performance metrics
        if event.was_correct:
            self.total_contradictions += 1
            if not event.false_positive:
                self.correct_detections += 1
        if event.false_positive:
            self.false_positives += 1
            
        # Store event
        self.events.append(event)
        if len(self.events) > self.max_buffer_size:
            self.events.pop(0)
            
        return old_gamma, new_gamma, reward
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for contradiction detection.
        """
        if self.total_contradictions == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive_rate': 0.0,
                'gamma': self.current_gamma
            }
            
        precision = self.correct_detections / (self.correct_detections + self.false_positives) if (self.correct_detections + self.false_positives) > 0 else 0
        recall = self.correct_detections / self.total_contradictions
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        false_positive_rate = self.false_positives / len(self.events) if self.events else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'gamma': self.current_gamma,
            'total_events': len(self.events)
        }
        
    def record_contradiction_event(self, token: str, memory_token: str, 
                                 similarity_score: float, contradiction_score: float,
                                 was_correct: bool, false_positive: bool):
        """
        Record a contradiction detection event for PPO training.
        """
        event = ContradictionEvent(
            timestamp=time.time(),
            gamma=self.current_gamma,
            token=token,
            memory_token=memory_token,
            similarity_score=similarity_score,
            contradiction_score=contradiction_score,
            was_correct=was_correct,
            false_positive=false_positive
        )
        
        old_gamma, new_gamma, reward = self.update_gamma(event)
        
        # Log significant changes
        if abs(new_gamma - old_gamma) > 0.01:
            print(f"🔄 Gamma updated: {old_gamma:.3f} → {new_gamma:.3f} (reward: {reward:.2f})")
            
        return event
        
    def save_training_history(self, filepath: str = "ppo_training_history.json"):
        """
        Save training history for analysis.
        """
        history = {
            'events': [
                {
                    'timestamp': event.timestamp,
                    'gamma': event.gamma,
                    'token': event.token,
                    'memory_token': event.memory_token,
                    'similarity_score': event.similarity_score,
                    'contradiction_score': event.contradiction_score,
                    'was_correct': event.was_correct,
                    'false_positive': event.false_positive
                }
                for event in self.events
            ],
            'performance_metrics': self.get_performance_metrics(),
            'config': {
                'learning_rate': self.learning_rate,
                'gamma_range': self.gamma_range,
                'clip_epsilon': self.clip_epsilon
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
            
    def load_training_history(self, filepath: str = "ppo_training_history.json"):
        """
        Load training history from file.
        """
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
                
            # Restore events
            self.events = []
            for event_data in history.get('events', []):
                event = ContradictionEvent(
                    timestamp=event_data['timestamp'],
                    gamma=event_data['gamma'],
                    token=event_data['token'],
                    memory_token=event_data['memory_token'],
                    similarity_score=event_data['similarity_score'],
                    contradiction_score=event_data['contradiction_score'],
                    was_correct=event_data['was_correct'],
                    false_positive=event_data['false_positive']
                )
                self.events.append(event)
                
            # Update gamma to latest value
            if self.events:
                self.current_gamma = self.events[-1].gamma
                
        except FileNotFoundError:
            print(f"⚠️  No training history found at {filepath}")
            
    def get_optimal_gamma(self) -> float:
        """
        Return the current gamma value, which is continuously optimized.
        """
        return self.current_gamma 