#!/usr/bin/env python3
"""
StrategyOptimizer - Strategy performance analysis and optimization

Analyzes the performance of different repair strategies and provides
optimization recommendations based on historical data.

Features:
- Strategy performance analysis
- Success rate calculation
- Average reflex score tracking
- Strategy recommendation generation
- Performance trend analysis
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from storage.reflex_log import get_reflex_logger
from config.settings import get_config


@dataclass
class StrategyPerformance:
    """Represents performance metrics for a strategy."""
    strategy: str
    total_executions: int
    successful_executions: int
    average_score: float
    success_rate: float
    best_score: float
    worst_score: float
    recent_trend: float  # Average of last 10 scores
    last_updated: float


class StrategyOptimizer:
    """
    Analyzes strategy performance and provides optimization recommendations.
    
    Features:
    - Strategy performance analysis
    - Success rate calculation
    - Average reflex score tracking
    - Strategy recommendation generation
    - Performance trend analysis
    """
    
    def __init__(self):
        self.reflex_logger = get_reflex_logger()
        self.config = get_config()
        
        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.last_analysis_time = 0
        self.analysis_interval = self.config.get('strategy_analysis_interval', 3600)  # 1 hour
        
        # Output file
        self.output_file = Path("data/strategy_performance.jsonl")
        
        print(f"[StrategyOptimizer] Initialized with analysis interval={self.analysis_interval}s")
    
    def analyze_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """
        Analyze performance of all strategies.
        
        Returns:
            Dictionary of strategy performance metrics
        """
        try:
            print(f"[StrategyOptimizer] Analyzing strategy performance...")
            
            # Get all reflex scores
            all_scores = self.reflex_logger.get_reflex_scores(limit=1000)
            
            # Group scores by strategy
            strategy_scores = {}
            for score in all_scores:
                strategy = score.strategy
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(score)
            
            # Calculate performance metrics for each strategy
            for strategy, scores in strategy_scores.items():
                performance = self._calculate_strategy_performance(strategy, scores)
                self.strategy_performance[strategy] = performance
            
            # Write results to file
            self._write_performance_results()
            
            self.last_analysis_time = time.time()
            print(f"[StrategyOptimizer] Analyzed {len(strategy_scores)} strategies")
            
            return self.strategy_performance
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error analyzing strategy performance: {e}")
            return {}
    
    def _calculate_strategy_performance(self, strategy: str, scores: List) -> StrategyPerformance:
        """
        Calculate performance metrics for a specific strategy.
        
        Args:
            strategy: Strategy name
            scores: List of reflex scores for this strategy
            
        Returns:
            StrategyPerformance object
        """
        try:
            total_executions = len(scores)
            successful_executions = sum(1 for s in scores if s.success)
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            
            # Calculate score metrics
            score_values = [s.score for s in scores]
            average_score = sum(score_values) / len(score_values) if score_values else 0.0
            best_score = max(score_values) if score_values else 0.0
            worst_score = min(score_values) if score_values else 0.0
            
            # Calculate recent trend (last 10 scores)
            recent_scores = score_values[:10]
            recent_trend = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            
            return StrategyPerformance(
                strategy=strategy,
                total_executions=total_executions,
                successful_executions=successful_executions,
                average_score=average_score,
                success_rate=success_rate,
                best_score=best_score,
                worst_score=worst_score,
                recent_trend=recent_trend,
                last_updated=time.time()
            )
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error calculating performance for {strategy}: {e}")
            return StrategyPerformance(
                strategy=strategy,
                total_executions=0,
                successful_executions=0,
                average_score=0.0,
                success_rate=0.0,
                best_score=0.0,
                worst_score=0.0,
                recent_trend=0.0,
                last_updated=time.time()
            )
    
    def get_best_strategies(self, limit: int = 5) -> List[StrategyPerformance]:
        """
        Get the best performing strategies.
        
        Args:
            limit: Maximum number of strategies to return
            
        Returns:
            List of best performing strategies
        """
        try:
            # Ensure we have fresh data
            if time.time() - self.last_analysis_time > self.analysis_interval:
                self.analyze_strategy_performance()
            
            # Sort by average score
            sorted_strategies = sorted(
                self.strategy_performance.values(),
                key=lambda x: x.average_score,
                reverse=True
            )
            
            return sorted_strategies[:limit]
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error getting best strategies: {e}")
            return []
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Generate strategy optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        try:
            recommendations = {
                "best_strategies": [],
                "improvement_areas": [],
                "trend_analysis": [],
                "suggestions": []
            }
            
            # Get best strategies
            best_strategies = self.get_best_strategies(3)
            recommendations["best_strategies"] = [
                {
                    "strategy": s.strategy,
                    "average_score": s.average_score,
                    "success_rate": s.success_rate,
                    "total_executions": s.total_executions
                }
                for s in best_strategies
            ]
            
            # Identify improvement areas
            for strategy, performance in self.strategy_performance.items():
                if performance.average_score < 0.6:
                    recommendations["improvement_areas"].append({
                        "strategy": strategy,
                        "current_score": performance.average_score,
                        "suggestion": "Consider alternative approaches or parameter tuning"
                    })
                
                # Check for declining trends
                if performance.recent_trend < performance.average_score * 0.8:
                    recommendations["trend_analysis"].append({
                        "strategy": strategy,
                        "trend": "declining",
                        "recent_average": performance.recent_trend,
                        "overall_average": performance.average_score
                    })
            
            # Generate suggestions
            if best_strategies:
                best_strategy = best_strategies[0]
                recommendations["suggestions"].append(
                    f"Prioritize {best_strategy.strategy} for similar drift patterns "
                    f"(avg score: {best_strategy.average_score:.3f})"
                )
            
            return recommendations
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error generating recommendations: {e}")
            return {}
    
    def _write_performance_results(self):
        """Write performance results to JSONL file."""
        try:
            with open(self.output_file, 'w') as f:
                for strategy, performance in self.strategy_performance.items():
                    result = {
                        "timestamp": time.time(),
                        "strategy": strategy,
                        "total_executions": performance.total_executions,
                        "successful_executions": performance.successful_executions,
                        "average_score": performance.average_score,
                        "success_rate": performance.success_rate,
                        "best_score": performance.best_score,
                        "worst_score": performance.worst_score,
                        "recent_trend": performance.recent_trend
                    }
                    f.write(json.dumps(result) + '\n')
            
            print(f"[StrategyOptimizer] Wrote performance results to {self.output_file}")
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error writing performance results: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dictionary with performance summary
        """
        try:
            if not self.strategy_performance:
                self.analyze_strategy_performance()
            
            total_executions = sum(p.total_executions for p in self.strategy_performance.values())
            overall_success_rate = sum(p.successful_executions for p in self.strategy_performance.values()) / total_executions if total_executions > 0 else 0.0
            overall_average_score = sum(p.average_score * p.total_executions for p in self.strategy_performance.values()) / total_executions if total_executions > 0 else 0.0
            
            return {
                "total_strategies": len(self.strategy_performance),
                "total_executions": total_executions,
                "overall_success_rate": overall_success_rate,
                "overall_average_score": overall_average_score,
                "last_analysis": self.last_analysis_time,
                "strategies": {
                    strategy: {
                        "total_executions": perf.total_executions,
                        "average_score": perf.average_score,
                        "success_rate": perf.success_rate
                    }
                    for strategy, perf in self.strategy_performance.items()
                }
            }
            
        except Exception as e:
            logging.error(f"[StrategyOptimizer] Error getting performance summary: {e}")
            return {}


# Global strategy optimizer instance
_strategy_optimizer_instance = None


def get_strategy_optimizer() -> StrategyOptimizer:
    """Get or create the global strategy optimizer instance."""
    global _strategy_optimizer_instance
    
    if _strategy_optimizer_instance is None:
        _strategy_optimizer_instance = StrategyOptimizer()
    
    return _strategy_optimizer_instance


def analyze_strategy_performance() -> Dict[str, Any]:
    """Analyze strategy performance and return results."""
    optimizer = get_strategy_optimizer()
    return optimizer.analyze_strategy_performance()


def get_strategy_recommendations() -> Dict[str, Any]:
    """Get strategy optimization recommendations."""
    optimizer = get_strategy_optimizer()
    return optimizer.get_strategy_recommendations()


def get_performance_summary() -> Dict[str, Any]:
    """Get strategy performance summary."""
    optimizer = get_strategy_optimizer()
    return optimizer.get_performance_summary() 