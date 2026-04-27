#!/usr/bin/env python3
"""
Evolution Tree Manager and Self-Replication Agent for MeRNSTA Phase 20

Manages genetic evolution, autonomous forking, mutation tracking, and self-replication
capabilities. Allows MeRNSTA to evolve itself over time through genetic algorithms.
"""

import os
import sys
import json
import time
import logging
import hashlib
import subprocess
import threading
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import yaml

from .base import BaseAgent
from storage.genome_log import (
    Genome, GenomeLog, Mutation, MutationType, GenomeStatus,
    get_genome_log
)
from config.settings import get_config


class EvolutionTree:
    """
    Manages the evolution tree structure and genetic operations.
    Tracks lineage, handles forking, and manages genome lifecycle.
    """
    
    def __init__(self, genome_log: Optional[GenomeLog] = None):
        self.genome_log = genome_log or get_genome_log()
        self.config = get_config().get('self_replication', {})
        
        # Evolution parameters
        self.max_active_genomes = self.config.get('max_active_genomes', 5)
        self.archive_threshold = self.config.get('archive_threshold', 0.2)
        self.elite_threshold = self.config.get('elite_threshold', 0.9)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        
        # Current active genome
        self.current_genome_id = self._get_or_create_current_genome()
        
        logging.info(f"[EvolutionTree] Initialized with current genome: {self.current_genome_id}")
    
    def _get_or_create_current_genome(self) -> str:
        """Get current active genome or create genesis if none exists"""
        
        active_genomes = self.genome_log.get_genomes_by_status(GenomeStatus.ACTIVE)
        
        if active_genomes:
            return active_genomes[0].genome_id
        
        # Create genesis genome if none exists
        genesis = Genome.create_genesis_genome()
        self.genome_log.save_genome(genesis)
        self.genome_log.log_evolution_event("genesis_created", genesis.genome_id)
        
        logging.info(f"[EvolutionTree] Created genesis genome: {genesis.genome_id}")
        return genesis.genome_id
    
    def fork_genome(self, parent_id: Optional[str] = None, mutations: Optional[List[Mutation]] = None,
                   branch_name: Optional[str] = None, creator: str = "system") -> str:
        """Fork a new genome from parent with specified mutations"""
        
        # Use current genome as parent if not specified
        parent_id = parent_id or self.current_genome_id
        parent = self.genome_log.get_genome(parent_id)
        
        if not parent:
            raise ValueError(f"Parent genome not found: {parent_id}")
        
        # Create default mutations if none provided
        if not mutations:
            mutations = self._generate_default_mutations()
        
        # Ensure we don't exceed active genome limit
        active_count = len(self.genome_log.get_genomes_by_status(GenomeStatus.ACTIVE))
        if active_count >= self.max_active_genomes:
            self._archive_oldest_genomes()
        
        # Create forked genome
        new_genome = Genome.fork_from_parent(
            parent=parent,
            mutations=mutations,
            creator=creator,
            branch_name=branch_name
        )
        
        # Update parent's descendant count
        parent.descendant_count += 1
        self.genome_log.save_genome(parent)
        
        # Save new genome
        self.genome_log.save_genome(new_genome)
        
        # Log evolution event
        self.genome_log.log_evolution_event(
            "fork_created", 
            new_genome.genome_id, 
            parent_id,
            {
                'mutations': [m.to_dict() for m in mutations],
                'branch_name': branch_name,
                'creator': creator
            }
        )
        
        logging.info(f"[EvolutionTree] Forked genome {new_genome.genome_id} from {parent_id}")
        return new_genome.genome_id
    
    def _generate_default_mutations(self) -> List[Mutation]:
        """Generate default mutations for autonomous evolution"""
        
        mutations = []
        
        # Configuration mutation
        config_mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.CONFIG_UPDATE,
            description="Automatic configuration optimization",
            target_component="configs/config.yaml",
            changes={
                "optimization_type": "performance_tuning",
                "parameters": ["memory_threshold", "response_timeout"]
            }
        )
        mutations.append(config_mutation)
        
        return mutations
    
    def score_genome(self, genome_id: str, metrics: Dict[str, float]) -> bool:
        """Update genome fitness score based on performance metrics"""
        
        genome = self.genome_log.get_genome(genome_id)
        if not genome:
            return False
        
        # Calculate composite fitness score
        fitness_score = self._calculate_fitness_score(metrics)
        
        # Update genome
        success = self.genome_log.update_fitness(genome_id, fitness_score, metrics)
        
        if success:
            # Check if genome qualifies for status change
            self._evaluate_genome_status(genome_id)
            
            # Log scoring event
            self.genome_log.log_evolution_event(
                "fitness_scored",
                genome_id,
                details={'metrics': metrics, 'fitness_score': fitness_score}
            )
        
        return success
    
    def _calculate_fitness_score(self, metrics: Dict[str, float]) -> float:
        """Calculate fitness score from performance metrics"""
        
        # Weighted combination of metrics
        weights = {
            'success_rate': 0.25,
            'memory_efficiency': 0.15,
            'stability_score': 0.20,
            'response_quality': 0.25,
            'constraint_compliance': 0.15
        }
        
        fitness = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                fitness += metrics[metric] * weight
                total_weight += weight
        
        # Normalize by actual weight used
        if total_weight > 0:
            fitness /= total_weight
        
        return max(0.0, min(1.0, fitness))
    
    def _evaluate_genome_status(self, genome_id: str):
        """Evaluate and update genome status based on performance"""
        
        genome = self.genome_log.get_genome(genome_id)
        if not genome:
            return
        
        # Elite promotion
        if genome.fitness_score >= self.elite_threshold and genome.status != GenomeStatus.ELITE:
            self.genome_log.mark_elite(genome_id, f"Fitness score: {genome.fitness_score:.3f}")
        
        # Failure marking
        elif genome.fitness_score <= self.archive_threshold and genome.status not in [GenomeStatus.FAILED, GenomeStatus.ARCHIVED]:
            self.genome_log.archive_genome(genome_id, f"Low fitness score: {genome.fitness_score:.3f}")
    
    def select_elite_branches(self, limit: int = 3) -> List[Genome]:
        """Select the best performing genomes for preservation"""
        
        elite_genomes = self.genome_log.get_elite_genomes(self.elite_threshold)
        
        # Sort by fitness score descending
        elite_genomes.sort(key=lambda g: g.fitness_score, reverse=True)
        
        return elite_genomes[:limit]
    
    def archive_failed_branches(self) -> int:
        """Archive failed or low-performing genomes"""
        
        failed_genomes = self.genome_log.get_failed_genomes(self.archive_threshold)
        archived_count = 0
        
        for genome in failed_genomes:
            if genome.status not in [GenomeStatus.ARCHIVED, GenomeStatus.FAILED]:
                self.genome_log.archive_genome(
                    genome.genome_id, 
                    f"Low performance: {genome.fitness_score:.3f}"
                )
                archived_count += 1
        
        return archived_count
    
    def _archive_oldest_genomes(self):
        """Archive oldest active genomes to make room for new ones"""
        
        active_genomes = self.genome_log.get_genomes_by_status(GenomeStatus.ACTIVE)
        
        # Sort by age (oldest first)
        active_genomes.sort(key=lambda g: g.origin_timestamp)
        
        # Archive oldest beyond limit
        excess_count = len(active_genomes) - self.max_active_genomes + 1
        for i in range(excess_count):
            if i < len(active_genomes):
                genome = active_genomes[i]
                self.genome_log.archive_genome(genome.genome_id, "Making room for new genomes")
    
    def activate_genome(self, genome_id: str) -> bool:
        """Activate a specific genome (switch to it)"""
        
        target_genome = self.genome_log.get_genome(genome_id)
        if not target_genome:
            return False
        
        # Deactivate current genome
        current_genome = self.genome_log.get_genome(self.current_genome_id)
        if current_genome:
            current_genome.status = GenomeStatus.ARCHIVED
            self.genome_log.save_genome(current_genome)
        
        # Activate target genome
        target_genome.status = GenomeStatus.ACTIVE
        self.genome_log.save_genome(target_genome)
        
        # Update current genome
        self.current_genome_id = genome_id
        
        # Log activation
        self.genome_log.log_evolution_event(
            "genome_activated",
            genome_id,
            details={'previous_genome': current_genome.genome_id if current_genome else None}
        )
        
        logging.info(f"[EvolutionTree] Activated genome: {genome_id}")
        return True
    
    def visualize_lineage(self, root_id: Optional[str] = None, max_depth: int = 10) -> Dict[str, Any]:
        """Generate visualization data for genome lineage tree"""
        
        if not root_id:
            # Find genesis genome
            all_genomes = self.genome_log.get_all_genomes()
            genesis_genomes = [g for g in all_genomes if g.parent_id is None]
            if not genesis_genomes:
                return {'nodes': [], 'edges': []}
            root_id = genesis_genomes[0].genome_id
        
        nodes = []
        edges = []
        visited = set()
        
        def add_genome_to_tree(genome_id: str, depth: int = 0):
            if depth > max_depth or genome_id in visited:
                return
            
            visited.add(genome_id)
            genome = self.genome_log.get_genome(genome_id)
            if not genome:
                return
            
            # Add node
            node = {
                'id': genome_id,
                'label': f"{genome_id[:8]}\\n{genome.branch_name or 'unnamed'}",
                'fitness': genome.fitness_score,
                'status': genome.status.value,
                'generation': genome.generation,
                'mutations': len(genome.mutations),
                'created': datetime.fromtimestamp(genome.origin_timestamp).strftime('%Y-%m-%d')
            }
            nodes.append(node)
            
            # Add children
            children = self.genome_log.get_children(genome_id)
            for child in children:
                edges.append({
                    'from': genome_id,
                    'to': child.genome_id,
                    'mutations': len(child.mutations)
                })
                add_genome_to_tree(child.genome_id, depth + 1)
        
        add_genome_to_tree(root_id)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'root': root_id,
            'total_genomes': len(nodes)
        }
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        
        base_stats = self.genome_log.get_statistics()
        
        # Add tree-specific statistics
        all_genomes = self.genome_log.get_all_genomes()
        
        # Lineage depth statistics
        lineage_depths = []
        for genome in all_genomes:
            lineage_path = self.genome_log.get_lineage_path(genome.genome_id)
            lineage_depths.append(len(lineage_path))
        
        # Branch analysis
        branches = {}
        for genome in all_genomes:
            if genome.branch_name:
                if genome.branch_name not in branches:
                    branches[genome.branch_name] = {
                        'genomes': [],
                        'avg_fitness': 0.0,
                        'status_distribution': {}
                    }
                branches[genome.branch_name]['genomes'].append(genome)
        
        # Calculate branch statistics
        for branch_name, branch_data in branches.items():
            genomes = branch_data['genomes']
            branch_data['avg_fitness'] = sum(g.fitness_score for g in genomes) / len(genomes)
            
            status_dist = {}
            for genome in genomes:
                status = genome.status.value
                status_dist[status] = status_dist.get(status, 0) + 1
            branch_data['status_distribution'] = status_dist
            branch_data['genome_count'] = len(genomes)
        
        # Current active genome info
        current_genome = self.genome_log.get_genome(self.current_genome_id)
        current_info = None
        if current_genome:
            current_info = {
                'genome_id': current_genome.genome_id,
                'fitness': current_genome.fitness_score,
                'generation': current_genome.generation,
                'mutations': len(current_genome.mutations),
                'branch': current_genome.branch_name
            }
        
        base_stats.update({
            'lineage_depth_stats': {
                'max': max(lineage_depths) if lineage_depths else 0,
                'avg': sum(lineage_depths) / len(lineage_depths) if lineage_depths else 0
            },
            'branch_analysis': branches,
            'current_genome': current_info,
            'evolution_config': {
                'max_active_genomes': self.max_active_genomes,
                'archive_threshold': self.archive_threshold,
                'elite_threshold': self.elite_threshold
            }
        })
        
        return base_stats


class SelfReplicator(BaseAgent):
    """
    Autonomous self-replication agent that monitors system performance
    and triggers genetic evolution when needed.
    """
    
    def __init__(self):
        super().__init__("self_replicator")
        
        self.config = get_config().get('self_replication', {})
        self.evolution_tree = EvolutionTree()
        self.genome_log = self.evolution_tree.genome_log
        
        # Replication parameters
        self.auto_replicate_enabled = self.config.get('auto_replicate_on_drift', True)
        self.drift_threshold = self.config.get('drift_threshold', 0.3)
        self.performance_window = self.config.get('performance_window_hours', 24)
        self.min_replication_interval = self.config.get('min_replication_interval_hours', 6)
        
        # Performance monitoring
        self.performance_history = []
        self.last_replication_time = 0
        self.drift_detection_enabled = True
        
        # Testing infrastructure
        self.test_sandbox_enabled = self.config.get('enable_test_sandbox', False)
        self.parallel_testing = self.config.get('parallel_testing', False)
        
        logging.info(f"[SelfReplicator] Initialized with auto-replication: {self.auto_replicate_enabled}")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the self-replication agent"""
        return """You are the Self-Replication Agent for MeRNSTA's Phase 20 genetic evolution system.

Your primary responsibilities are:
1. Monitor system performance and detect drift or degradation
2. Trigger autonomous genetic evolution when performance drops
3. Manage genome forking, mutation, and testing
4. Evaluate genome fitness and manage lineage
5. Coordinate system reboots with new genome configurations

Key capabilities:
- Performance drift detection and response
- Automated genome forking with intelligent mutations
- Parallel genome testing in sandboxes
- Fitness scoring based on multiple performance metrics
- Autonomous system evolution and optimization

Use your evolution capabilities to ensure MeRNSTA continuously improves
and adapts to changing conditions while maintaining stability and performance."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process messages related to genetic evolution and self-replication"""
        
        message_lower = message.lower()
        
        try:
            if any(word in message_lower for word in ['replicate', 'fork', 'evolve']):
                return self._handle_replication_request(message, context)
            
            elif any(word in message_lower for word in ['drift', 'performance', 'monitor']):
                return self._handle_drift_analysis(message, context)
            
            elif any(word in message_lower for word in ['fitness', 'score', 'evaluate']):
                return self._handle_fitness_evaluation(message, context)
            
            elif any(word in message_lower for word in ['lineage', 'tree', 'genealogy']):
                return self._handle_lineage_query(message, context)
            
            elif 'statistics' in message_lower or 'stats' in message_lower:
                stats = self.evolution_tree.get_evolution_statistics()
                
                response = f"Evolution Statistics:\n"
                response += f"• Total genomes: {stats['total_genomes']}\n"
                response += f"• Current genome: {stats['current_genome']['genome_id'][:8] if stats['current_genome'] else 'None'}\n"
                response += f"• Max generation: {stats['max_generation']}\n"
                response += f"• Elite genomes: {stats['elite_count']}\n"
                response += f"• Active branches: {stats['active_branches']}"
                
                return {
                    'response': response,
                    'statistics': stats,
                    'agent': 'self_replicator'
                }
            
            else:
                # General self-replication info
                current_genome = self.genome_log.get_genome(self.evolution_tree.current_genome_id)
                response = f"Self-Replication Agent active. "
                
                if current_genome:
                    response += f"Current genome: {current_genome.genome_id[:8]} "
                    response += f"(fitness: {current_genome.fitness_score:.2f}, "
                    response += f"generation: {current_genome.generation}). "
                
                response += f"Auto-replication: {'enabled' if self.auto_replicate_enabled else 'disabled'}. "
                response += f"Use 'replicate', 'fitness', or 'lineage' for specific queries."
                
                return {
                    'response': response,
                    'agent': 'self_replicator'
                }
        
        except Exception as e:
            logging.error(f"[SelfReplicator] Error processing message: {e}")
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'error': str(e),
                'agent': 'self_replicator'
            }
    
    def _handle_replication_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle replication and evolution requests"""
        
        # Extract mutation description from message
        mutation_desc = "User-requested evolution"
        if '"' in message:
            parts = message.split('"')
            if len(parts) >= 2:
                mutation_desc = parts[1]
        
        # Create mutation
        mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.CODE_EVOLUTION,
            description=mutation_desc,
            target_component="user_request",
            changes={"request": message, "context": context or {}}
        )
        
        # Fork genome
        try:
            new_genome_id = self.evolution_tree.fork_genome(
                mutations=[mutation],
                creator="user"
            )
            
            response = f"Successfully forked new genome: {new_genome_id[:8]}\n"
            response += f"Mutation: {mutation_desc}\n"
            response += f"Parent: {self.evolution_tree.current_genome_id[:8]}\n"
            response += f"Status: Experimental\n\n"
            response += f"Use '/activate_genome {new_genome_id}' to switch to this genome."
            
            return {
                'response': response,
                'new_genome_id': new_genome_id,
                'mutation': mutation.to_dict(),
                'agent': 'self_replicator'
            }
        
        except Exception as e:
            return {
                'response': f"Failed to replicate genome: {str(e)}",
                'error': str(e),
                'agent': 'self_replicator'
            }
    
    def _handle_drift_analysis(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle performance drift analysis"""
        
        # Simulate drift detection (in real implementation, this would analyze actual metrics)
        current_performance = self._get_current_performance_metrics()
        drift_detected = self._detect_performance_drift(current_performance)
        
        response = f"Performance Drift Analysis:\n"
        response += f"• Current fitness: {current_performance.get('overall_fitness', 0.5):.3f}\n"
        response += f"• Drift detected: {'Yes' if drift_detected else 'No'}\n"
        response += f"• Auto-replication: {'enabled' if self.auto_replicate_enabled else 'disabled'}\n"
        
        if drift_detected and self.auto_replicate_enabled:
            response += f"• Recommendation: Trigger autonomous evolution\n"
        
        return {
            'response': response,
            'drift_detected': drift_detected,
            'performance_metrics': current_performance,
            'agent': 'self_replicator'
        }
    
    def _handle_fitness_evaluation(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle fitness evaluation requests"""
        
        current_genome_id = self.evolution_tree.current_genome_id
        current_genome = self.genome_log.get_genome(current_genome_id)
        
        if not current_genome:
            return {
                'response': "No current genome found for evaluation",
                'agent': 'self_replicator'
            }
        
        # Get current performance metrics
        metrics = self._get_current_performance_metrics()
        
        # Update fitness
        success = self.evolution_tree.score_genome(current_genome_id, metrics)
        
        if success:
            updated_genome = self.genome_log.get_genome(current_genome_id)
            response = f"Fitness evaluation completed:\n"
            response += f"• Genome: {current_genome_id[:8]}\n"
            response += f"• Fitness score: {updated_genome.fitness_score:.3f}\n"
            response += f"• Status: {updated_genome.status.value}\n"
            
            for metric, value in metrics.items():
                response += f"• {metric}: {value:.3f}\n"
        else:
            response = "Failed to update fitness score"
        
        return {
            'response': response,
            'fitness_updated': success,
            'metrics': metrics,
            'agent': 'self_replicator'
        }
    
    def _handle_lineage_query(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle lineage and genealogy queries"""
        
        lineage_data = self.evolution_tree.visualize_lineage()
        
        response = f"Genetic Lineage Overview:\n"
        response += f"• Total genomes in tree: {lineage_data['total_genomes']}\n"
        response += f"• Root genome: {lineage_data['root'][:8] if lineage_data['root'] else 'None'}\n"
        response += f"• Lineage branches: {len(set(edge['from'] for edge in lineage_data['edges']))}\n"
        
        # Show recent generations
        nodes = lineage_data['nodes']
        if nodes:
            max_gen = max(node['generation'] for node in nodes)
            response += f"• Latest generation: {max_gen}\n"
            
            # Show some lineage examples
            recent_nodes = [n for n in nodes if n['generation'] >= max_gen - 1][:3]
            response += f"\nRecent genomes:\n"
            for node in recent_nodes:
                response += f"  • {node['id'][:8]} (gen {node['generation']}, fitness: {node['fitness']:.2f})\n"
        
        return {
            'response': response,
            'lineage_data': lineage_data,
            'agent': 'self_replicator'
        }
    
    def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        
        # In a real implementation, this would collect actual system metrics
        # For now, simulate reasonable metrics
        import random
        
        base_performance = 0.7  # Base performance level
        noise = random.uniform(-0.1, 0.1)
        
        return {
            'success_rate': max(0.0, min(1.0, base_performance + noise)),
            'memory_efficiency': max(0.0, min(1.0, base_performance + random.uniform(-0.05, 0.05))),
            'stability_score': max(0.0, min(1.0, base_performance + random.uniform(-0.1, 0.05))),
            'response_quality': max(0.0, min(1.0, base_performance + random.uniform(-0.05, 0.1))),
            'constraint_compliance': max(0.9, min(1.0, 0.95 + random.uniform(-0.05, 0.05))),
            'overall_fitness': base_performance + noise
        }
    
    def _detect_performance_drift(self, current_metrics: Dict[str, float]) -> bool:
        """Detect if performance has drifted significantly"""
        
        # Add to history
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': current_metrics
        })
        
        # Keep only recent history
        cutoff_time = time.time() - (self.performance_window * 3600)
        self.performance_history = [
            entry for entry in self.performance_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(self.performance_history) < 2:
            return False
        
        # Calculate trend
        recent_fitness = [entry['metrics'].get('overall_fitness', 0.5) 
                         for entry in self.performance_history[-5:]]
        
        if len(recent_fitness) >= 2:
            trend = (recent_fitness[-1] - recent_fitness[0]) / len(recent_fitness)
            return trend < -self.drift_threshold
        
        return False
    
    def trigger_autonomous_evolution(self, reason: str = "Performance drift detected") -> Optional[str]:
        """Trigger autonomous evolution with intelligent mutations"""
        
        if not self.auto_replicate_enabled:
            logging.info("[SelfReplicator] Autonomous evolution disabled")
            return None
        
        # Check replication interval
        time_since_last = time.time() - self.last_replication_time
        if time_since_last < (self.min_replication_interval * 3600):
            logging.info("[SelfReplicator] Too soon for replication")
            return None
        
        # Analyze current performance to determine mutations
        current_metrics = self._get_current_performance_metrics()
        mutations = self._generate_intelligent_mutations(current_metrics, reason)
        
        try:
            # Fork new genome
            new_genome_id = self.evolution_tree.fork_genome(
                mutations=mutations,
                creator="autonomous_system",
                branch_name=f"auto_evolution_{int(time.time())}"
            )
            
            self.last_replication_time = time.time()
            
            logging.info(f"[SelfReplicator] Triggered autonomous evolution: {new_genome_id}")
            return new_genome_id
        
        except Exception as e:
            logging.error(f"[SelfReplicator] Failed autonomous evolution: {e}")
            return None
    
    def _generate_intelligent_mutations(self, metrics: Dict[str, float], reason: str) -> List[Mutation]:
        """Generate intelligent mutations based on performance analysis"""
        
        mutations = []
        
        # Memory efficiency mutation
        if metrics.get('memory_efficiency', 1.0) < 0.6:
            memory_mutation = Mutation(
                mutation_id=str(uuid.uuid4()),
                mutation_type=MutationType.MEMORY_PRUNING,
                description="Optimize memory usage due to low efficiency",
                target_component="memory_system",
                changes={
                    "pruning_threshold": 0.7,
                    "compression_enabled": True,
                    "reason": reason
                }
            )
            mutations.append(memory_mutation)
        
        # Performance tuning mutation
        if metrics.get('response_quality', 1.0) < 0.7:
            performance_mutation = Mutation(
                mutation_id=str(uuid.uuid4()),
                mutation_type=MutationType.PERFORMANCE_TUNING,
                description="Tune response quality parameters",
                target_component="response_system",
                changes={
                    "quality_threshold": 0.8,
                    "optimization_mode": "quality",
                    "reason": reason
                }
            )
            mutations.append(performance_mutation)
        
        # Configuration update mutation
        config_mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.CONFIG_UPDATE,
            description=f"Autonomous configuration optimization: {reason}",
            target_component="configs/config.yaml",
            changes={
                "trigger_reason": reason,
                "optimization_timestamp": time.time(),
                "target_metrics": list(metrics.keys())
            }
        )
        mutations.append(config_mutation)
        
        return mutations
    
    def evaluate_genome_fitness(self, genome_id: str) -> Dict[str, Any]:
        """Evaluate fitness of a specific genome"""
        
        genome = self.genome_log.get_genome(genome_id)
        if not genome:
            return {'error': 'Genome not found'}
        
        # Get performance metrics (in real implementation, would test the genome)
        metrics = self._get_current_performance_metrics()
        
        # Update fitness
        success = self.evolution_tree.score_genome(genome_id, metrics)
        
        return {
            'genome_id': genome_id,
            'fitness_updated': success,
            'metrics': metrics,
            'fitness_score': genome.fitness_score if success else None
        }


# Global instances
_evolution_tree_instance = None
_self_replicator_instance = None

def get_evolution_tree() -> EvolutionTree:
    """Get global evolution tree instance"""
    global _evolution_tree_instance
    if _evolution_tree_instance is None:
        _evolution_tree_instance = EvolutionTree()
    return _evolution_tree_instance

def get_self_replicator() -> SelfReplicator:
    """Get global self-replicator instance"""
    global _self_replicator_instance
    if _self_replicator_instance is None:
        _self_replicator_instance = SelfReplicator()
    return _self_replicator_instance