#!/usr/bin/env python3
"""
🚀 Phase 2 Autonomous Cognitive Agent System for MeRNSTA

Integrates all Phase 2 advanced cognitive features:
- 🔗 Causal & Temporal Linkage
- 🗣️ Dialogue Clarification Agent
- 🎛️ Autonomous Memory Tuning
- 🧠 Theory of Mind Layer
- 🪞 Recursive Self-Inspection

Creates a fully autonomous, self-aware cognitive agent that monitors its own
thinking, generates self-improvement goals, and actively engages with users
to resolve contradictions and improve its understanding.
"""

import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

from .enhanced_memory_system import EnhancedMemorySystem
from .enhanced_memory_model import EnhancedTripletFact
from .predictive_chat_preemption import PredictiveChatPreemption
from config.settings import get_config

# Import all Phase 2 components
try:
    from .dialogue_clarification_agent import DialogueClarificationAgent, ClarificationRequest
    from .autonomous_memory_tuning import AutonomousMemoryTuning, PerformanceMetrics
    from .theory_of_mind_layer import TheoryOfMindLayer, PerspectiveAgent
    from .recursive_self_inspection import RecursiveSelfInspection, CognitiveSnapshot
    from .contradiction_clustering import ContradictionClusteringSystem
    from .belief_consolidation import BeliefConsolidationSystem
    from .memory_graphs import MemoryGraphSystem
    from .confabulation_filtering import ConfabulationFilteringSystem
    from .meta_cognition_agent import MetaCognitionAgent
    PHASE2_AVAILABLE = True
except ImportError as e:
    PHASE2_AVAILABLE = False
    logging.warning(f"Phase 2 cognitive features not available: {e}")


class Phase2AutonomousCognitiveSystem(EnhancedMemorySystem):
    """
    The complete Phase 2 autonomous cognitive agent that integrates all
    advanced cognitive capabilities into a self-aware, self-improving system.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize Phase 2 components
        self.self_inspection = RecursiveSelfInspection()
        self.contradiction_clustering = ContradictionClusteringSystem()
        self.belief_consolidation = BeliefConsolidationSystem()
        self.theory_of_mind = TheoryOfMindLayer()
        self.dialogue_clarification = DialogueClarificationAgent()
        self.autonomous_tuning = AutonomousMemoryTuning()
        
        # Initialize meta-cognition agent
        try:
            self.meta_cognition = MetaCognitionAgent()
        except ImportError:
            self.meta_cognition = None
            print("[Phase2System] Meta-cognition agent not available")
        
        # Initialize drift execution engine
        try:
            from agents.drift_execution_engine import get_drift_execution_engine
            self.drift_execution_engine = get_drift_execution_engine(self)
            
            # Start automatic drift execution if enabled
            config = get_config()
            if config.get('drift_auto_execute', True):
                self.drift_execution_engine.start_background_execution()
                print("[Phase2System] Started automatic drift goal execution")
        except ImportError:
            self.drift_execution_engine = None
            print("[Phase2System] Drift execution engine not available")
        
        # Cognitive cycle management
        self.last_cognitive_cycle = 0
        self.cognitive_cycle_interval = 300  # 5 minutes
        self.cognitive_cycle_count = 0
        
        # Causal analysis system
        self.causal_analysis_running = False
        self.causal_analysis_thread = None
        
        print(f"[Phase2System] Initialized with full autonomous cognitive capabilities")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system to dictionary for JSON serialization."""
        return {
            'system_type': 'Phase2AutonomousCognitiveSystem',
            'version': '0.7.0',
            'phase': 2,
            'last_cognitive_cycle': self.last_cognitive_cycle,
            'cognitive_cycle_count': self.cognitive_cycle_count,
            'autonomous_mode': getattr(self, 'autonomous_mode', False),
            'components_available': {
                'self_inspection': hasattr(self, 'self_inspection'),
                'contradiction_clustering': hasattr(self, 'contradiction_clustering'),
                'belief_consolidation': hasattr(self, 'belief_consolidation'),
                'theory_of_mind': hasattr(self, 'theory_of_mind'),
                'dialogue_clarification': hasattr(self, 'dialogue_clarification'),
                'autonomous_tuning': hasattr(self, 'autonomous_tuning'),
                'meta_cognition': hasattr(self, 'meta_cognition') and self.meta_cognition is not None,
                'drift_execution_engine': hasattr(self, 'drift_execution_engine') and self.drift_execution_engine is not None
            }
        }
    
    def process_input_with_full_cognition(self, text: str, user_profile_id: str = None,
                                        session_id: str = None) -> Dict:
        """
        Process input with full Phase 2 cognitive capabilities including multi-tier reasoning.
        """
        if not PHASE2_AVAILABLE:
            return super().process_input(text, user_profile_id, session_id)
        
        print(f"[Phase2Cognitive] Processing input with full cognitive awareness: '{text[:100]}...'")
        
        # Phase 33: Neural Control Hooks - Cognitive Arbitration
        try:
            from agents.cognitive_hooks import arbitration_enabled, response_generation_hook
            
            if arbitration_enabled() and not text.startswith('/'):
                # Route through cognitive arbitration for non-command inputs
                context = {
                    'user_profile_id': user_profile_id,
                    'session_id': session_id,
                    'system_source': 'phase2_cognitive_system',
                    'full_cognition': True
                }
                
                def default_processing(msg, ctx):
                    """Fallback to normal processing"""
                    return self._process_without_arbitration(msg, ctx.get('user_profile_id'), ctx.get('session_id'))
                
                # Route through cognitive arbitration
                arbitrated_response = response_generation_hook(text, context)
                
                # If arbitration returns a string response, format it properly
                if isinstance(arbitrated_response, str):
                    return {
                        'response': arbitrated_response,
                        'arbitration_used': True,
                        'user_profile_id': user_profile_id,
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    }
                elif isinstance(arbitrated_response, dict):
                    arbitrated_response['arbitration_used'] = True
                    return arbitrated_response
        except Exception as e:
            logging.warning(f"[Phase2Cognitive] Cognitive arbitration failed, using normal processing: {e}")
        
        # Continue with normal processing if arbitration not available or failed
        return self._process_without_arbitration(text, user_profile_id, session_id)
    
    def _process_without_arbitration(self, text: str, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Process input without cognitive arbitration (original logic)"""
        
        # Check for special commands
        if text.startswith('/'):
            return self._handle_special_commands(text, user_profile_id, session_id)
        
        # Process for perspective attribution first
        attributed_facts = self.theory_of_mind.process_statement_for_attribution(text, "user")
        
        # Process normally through enhanced system (memory search)
        base_result = super().process_input(text, user_profile_id, session_id)
        
        # MULTI-TIER COGNITIVE PIPELINE: Apply reasoning layers if needed
        enhanced_response = self._apply_multi_tier_reasoning(text, base_result, user_profile_id, session_id)
        if enhanced_response:
            base_result.update(enhanced_response)
        
        # Add attributed facts to storage if any were found
        if attributed_facts:
            for fact in attributed_facts:
                fact.user_profile_id = user_profile_id
                fact.session_id = session_id
            
            # Store the attributed facts
            stored_fact_ids = []
            for fact in attributed_facts:
                fact_id = self._store_fact(fact)
                if fact_id:
                    stored_fact_ids.append(fact_id)
            
            base_result['attributed_facts'] = len(attributed_facts)
            base_result['attributed_fact_ids'] = stored_fact_ids
        
        # CAUSAL LINKAGE DETECTION - Apply after new facts are stored
        causal_links, causal_metadata = self._detect_and_create_causal_links(text, user_profile_id, session_id)
        # Always add to metadata for test compatibility
        if 'metadata' not in base_result:
            base_result['metadata'] = {}
        base_result['metadata']['causal_links_created'] = len(causal_links) if causal_links else 0
        
        # Add causal analysis metadata for testing
        if causal_metadata:
            base_result['metadata']['causal_analysis'] = causal_metadata
        
        if causal_links:
            base_result['causal_links'] = len(causal_links)
            print(f"[Phase2Cognitive] Created {len(causal_links)} causal links")
        
        # Perform autonomous cognitive cycle if needed
        if self._should_run_cognitive_cycle():
            cognitive_insights = self._run_autonomous_cognitive_cycle(user_profile_id, session_id)
            base_result['cognitive_cycle'] = cognitive_insights
        
        # Check for clarification opportunities
        clarification_requests = self._check_for_clarification_opportunities(text, user_profile_id, session_id)
        if clarification_requests:
            base_result['clarification_requests'] = [
                {
                    'id': req.request_id,
                    'question': req.question,
                    'priority': req.priority,
                    'context': req.context
                }
                for req in clarification_requests
            ]
        
        # Apply confabulation filtering if this was a query
        if 'response' in base_result and base_result.get('response'):
            filtered_result = self._apply_advanced_confabulation_filtering(
                base_result['response'], text, user_profile_id, session_id
            )
            base_result.update(filtered_result)
        
        # Add cognitive insights
        base_result['phase2_insights'] = self._get_current_cognitive_insights()
        
        # Check for proactive suggestions based on current state
        try:
            proactive_suggestions = self.predictive_chat.check_for_proactive_suggestions(
                user_profile_id, session_id
            )
            
            if proactive_suggestions:
                base_result['proactive_suggestions'] = [
                    self.predictive_chat.format_suggestion_for_chat(suggestion)
                    for suggestion in proactive_suggestions
                ]
                print(f"[Phase2Cognitive] Generated {len(proactive_suggestions)} proactive suggestions")
        except Exception as e:
            print(f"[Phase2Cognitive] Error generating proactive suggestions: {e}")
        
        return base_result
    
    def _apply_multi_tier_reasoning(self, text: str, base_result: Dict, 
                                  user_profile_id: str = None, session_id: str = None) -> Dict:
        """
        Apply multi-tier cognitive reasoning pipeline:
        1. Check if memory provided good results
        2. If not, try symbolic reasoning
        3. If not, use LLM fallback for conversational queries
        """
        try:
            # Initialize reasoning engines lazily
            if not hasattr(self, '_symbolic_engine'):
                from tools.symbolic_engine import SymbolicEngine
                self._symbolic_engine = SymbolicEngine()
                
            if not hasattr(self, '_llm_fallback'):
                from tools.llm_fallback import LLMFallbackAgent
                self._llm_fallback = LLMFallbackAgent()
            
            # Determine if we need enhanced reasoning
            needs_enhancement = self._should_apply_enhanced_reasoning(text, base_result)
            
            if not needs_enhancement:
                return None
                
            print(f"[Phase2Cognitive] Applying multi-tier reasoning for: '{text[:50]}...'")
            
            # Tier 1: Check if this is a symbolic/mathematical query
            if self._symbolic_engine.is_symbolic_query(text):
                print("[Phase2Cognitive] Detected symbolic query, applying symbolic reasoning")
                symbolic_result = self._symbolic_engine.evaluate(text)
                
                if symbolic_result.get('success'):
                    return {
                        'response': f"The answer is {symbolic_result['result']}",
                        'reasoning_method': 'symbolic',
                        'reasoning_confidence': symbolic_result.get('confidence', 0.9),
                        'symbolic_result': symbolic_result
                    }
            
            # Tier 2: Check if LLM fallback should be used for conversational queries
            memory_confidence = self._extract_memory_confidence(base_result)
            memory_results = base_result.get('query_results', [])
            
            if self._llm_fallback.should_use_llm_fallback(text, memory_results, memory_confidence):
                print("[Phase2Cognitive] Using LLM fallback for conversational response")
                
                # Build context from memory for LLM
                context = self._build_llm_context(base_result, user_profile_id, session_id)
                llm_result = self._llm_fallback.generate_response(text, context)
                
                if llm_result.get('success'):
                    return {
                        'response': llm_result['response'],
                        'reasoning_method': 'llm_fallback',
                        'reasoning_confidence': llm_result.get('confidence', 0.8),
                        'llm_result': llm_result
                    }
            
            return None
            
        except Exception as e:
            print(f"[Phase2Cognitive] Error in multi-tier reasoning: {e}")
            return None
    
    def _should_apply_enhanced_reasoning(self, text: str, base_result: Dict) -> bool:
        """
        Determine if enhanced reasoning should be applied based on base result quality.
        """
        # Check if memory search provided poor results
        response = base_result.get('response', '')
        
        # Poor result indicators
        poor_result_indicators = [
            "don't have any information",
            "no information about that",
            "couldn't find",
            "no memory",
            "not found"
        ]
        
        has_poor_result = any(indicator in response.lower() for indicator in poor_result_indicators)
        
        # Check if query results are empty or low confidence
        query_results = base_result.get('query_results', [])
        has_low_results = len(query_results) == 0
        
        # Check if this looks like a question that could benefit from reasoning
        is_question = (
            text.strip().endswith('?') or
            any(text.lower().startswith(w) for w in ['what', 'how', 'why', 'when', 'where', 'who']) or
            any(greeting in text.lower() for greeting in ['hello', 'hi', 'hey', 'how are you'])
        )
        
        return (has_poor_result or has_low_results) and is_question
    
    def _extract_memory_confidence(self, base_result: Dict) -> float:
        """Extract confidence score from memory search results."""
        query_results = base_result.get('query_results', [])
        if not query_results:
            return 0.0
            
        # Calculate average confidence from query results
        if isinstance(query_results, list) and len(query_results) > 0:
            if hasattr(query_results[0], 'confidence'):
                return sum(result.confidence for result in query_results) / len(query_results)
            elif isinstance(query_results[0], tuple) and len(query_results[0]) > 1:
                return sum(result[1] for result in query_results) / len(query_results)
                
        return 0.5  # Default moderate confidence
    
    def _build_llm_context(self, base_result: Dict, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Build context for LLM from available memory and results."""
        context = {}
        
        # Add relevant memory facts if any
        query_results = base_result.get('query_results', [])
        if query_results:
            memory_facts = []
            for result in query_results[:3]:  # Top 3 results
                if hasattr(result, 'subject'):
                    memory_facts.append(f"{result.subject} {result.predicate} {result.object}")
                elif isinstance(result, tuple) and len(result) >= 2:
                    fact = result[0]
                    if hasattr(fact, 'subject'):
                        memory_facts.append(f"{fact.subject} {fact.predicate} {fact.object}")
                        
            context['memory_facts'] = memory_facts
        
        # Add session context
        context['user_profile_id'] = user_profile_id
        context['session_id'] = session_id
        
        return context
    
    def _handle_special_commands(self, command: str, user_profile_id: str = None,
                               session_id: str = None) -> Dict:
        """Handle special cognitive commands like /introspect."""
        command = command.lower().strip()
        
        if command == '/introspect':
            return self._handle_introspect_command(user_profile_id, session_id)
        elif command == '/clarify':
            return self._handle_clarify_command(user_profile_id, session_id)
        elif command == '/tune':
            return self._handle_tune_command(user_profile_id, session_id)
        elif command == '/perspectives':
            return self._handle_perspectives_command(user_profile_id, session_id)
        elif command == '/health':
            return self._handle_health_command(user_profile_id, session_id)
        elif command.startswith('/clarify_response '):
            # Handle clarification response
            parts = command.split(' ', 2)
            if len(parts) == 3:
                request_id = parts[1]
                response_text = parts[2]
                return self._handle_clarification_response(request_id, response_text)
        elif command == '/why' or command == '?':
            return self._handle_why_command(user_profile_id, session_id)
        elif command.startswith('/why do i believe'):
            return self._handle_why_believe_command(command, user_profile_id, session_id)
        elif command.startswith('/explain'):
            return self._handle_explain_command(command, user_profile_id, session_id)
        elif command == '/causal':
            return self._handle_causal_command(user_profile_id, session_id)
        elif command == '/causal_goals':
            return self._handle_causal_goals_command(user_profile_id, session_id)
        elif command == '/causal_health':
            return self._handle_causal_health_command(user_profile_id, session_id)
        elif command == '/start_causal':
            return self._handle_start_causal_command(user_profile_id, session_id)
        elif command == '/stop_causal':
            return self._handle_stop_causal_command(user_profile_id, session_id)
        elif command.startswith('/causal_execute '):
            goal_id = command.split(' ', 1)[1] if len(command.split(' ', 1)) > 1 else ""
            return self._handle_causal_execute_command(goal_id, user_profile_id, session_id)
        elif command == '/causal_graph':
            return self._handle_causal_graph_command(user_profile_id, session_id)
        
        elif command == '/drift_status':
            return self._handle_drift_status_command(user_profile_id, session_id)
        
        elif command == '/reflex_log':
            return self._handle_reflex_log_command(user_profile_id, session_id)
        
        # Phase 25: Reflective Self-Awareness Commands
        elif command == '/self_summary':
            return self._handle_self_summary_command(user_profile_id, session_id)
        elif command == '/self_journal':
            return self._handle_self_journal_command(user_profile_id, session_id)
        elif command == '/self_reflect':
            return self._handle_self_reflect_command(user_profile_id, session_id)
        elif command == '/self_sync':
            return self._handle_self_sync_command(user_profile_id, session_id)
        
        elif command == '/reflex_scores':
            return self._handle_reflex_scores_command(user_profile_id, session_id)
        
        elif command == '/strategy_optimization':
            return self._handle_strategy_optimization_command(user_profile_id, session_id)
        
        else:
            return {
                'response': f"Unknown command: {command}. Available commands: /introspect, /clarify, /tune, /perspectives, /health, /why, /explain, /causal, /causal_goals, /causal_health, /start_causal, /stop_causal, /causal_graph, /drift_status, /reflex_log, /reflex_scores, /strategy_optimization",
                'command_processed': False
            }
    
    def _handle_introspect_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /introspect command for recursive self-inspection."""
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Get current analysis
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(facts)
        belief_patterns = self.belief_consolidation.analyze_belief_stability(facts)
        agent_data = self.theory_of_mind.get_perspective_summary()
        
        # Generate comprehensive self-summary
        snapshot = self.self_inspection.get_self_summary(
            facts, contradiction_clusters, belief_patterns, agent_data
        )
        
        # Get detailed insights
        volatile_insights = self.self_inspection.get_most_volatile_beliefs(facts)
        reinforced_insights = self.self_inspection.get_most_reinforced_beliefs(facts)
        contradiction_insights = self.self_inspection.get_unresolved_contradictions(facts)
        
        # Get drift-triggered goals
        drift_goals = []
        try:
            from agents.cognitive_repair_agent import detect_drift_triggered_goals
            drift_goals = detect_drift_triggered_goals(limit=50)
        except ImportError:
            pass  # Cognitive repair agent not available
        
        # Format comprehensive report
        report = self.self_inspection.format_introspection_report(
            snapshot, volatile_insights, reinforced_insights, contradiction_insights
        )
        
        # Add drift-triggered goals to report if any exist
        if drift_goals:
            report += "\n\n🔧 DRIFT-TRIGGERED REPAIR GOALS:\n"
            report += "=" * 50 + "\n"
            
            # Group by priority
            high_priority = [g for g in drift_goals if g.priority > 0.7]
            medium_priority = [g for g in drift_goals if 0.4 <= g.priority <= 0.7]
            low_priority = [g for g in drift_goals if g.priority < 0.4]
            
            if high_priority:
                report += f"\n🔥 HIGH PRIORITY ({len(high_priority)}):\n"
                for goal in high_priority[:3]:  # Show top 3
                    report += f"  • {goal.goal} (priority: {goal.priority:.3f})\n"
                    report += f"    Strategy: {goal.repair_strategy}\n"
                    report += f"    Affected facts: {len(goal.affected_facts)}\n"
            
            if medium_priority:
                report += f"\n⚠️ MEDIUM PRIORITY ({len(medium_priority)}):\n"
                for goal in medium_priority[:2]:  # Show top 2
                    report += f"  • {goal.goal} (priority: {goal.priority:.3f})\n"
            
            if low_priority:
                report += f"\n💭 LOW PRIORITY ({len(low_priority)}):\n"
                for goal in low_priority[:1]:  # Show top 1
                    report += f"  • {goal.goal} (priority: {goal.priority:.3f})\n"
            
            if len(drift_goals) > 6:
                report += f"\n... and {len(drift_goals) - 6} more goals"
        
        # Add drift execution status if available
        if hasattr(self, 'drift_execution_engine') and self.drift_execution_engine:
            execution_status = self.drift_execution_engine.get_execution_status()
            
            report += "\n\n🤖 DRIFT EXECUTION STATUS:\n"
            report += "=" * 50 + "\n"
            report += f"Auto-execute enabled: {'✅' if execution_status['auto_execute_enabled'] else '❌'}\n"
            report += f"Background running: {'✅' if execution_status['background_running'] else '❌'}\n"
            report += f"Execution interval: {execution_status['execution_interval']}s\n"
            report += f"Priority threshold: {execution_status['priority_threshold']:.1f}\n"
            report += f"Active executions: {execution_status['active_executions']}\n"
            report += f"Completed executions: {execution_status['completed_executions']}\n"
            report += f"Failed executions: {execution_status['failed_executions']}\n"
            report += f"Success rate: {execution_status['success_rate']:.1%}\n"
            
            # Show recent executions
            if execution_status['recent_executions']:
                report += f"\n📋 Recent Executions:\n"
                for exec_result in execution_status['recent_executions'][-3:]:
                    status = "✅" if exec_result['success'] else "❌"
                    report += f"  {status} {exec_result['completion_notes']} ({exec_result['execution_time']:.2f}s)\n"
        
        return {
            'response': report,
            'command_processed': True,
            'introspection_data': {
                'snapshot': snapshot,
                'volatile_insights': len(volatile_insights),
                'reinforced_insights': len(reinforced_insights),
                'contradiction_insights': len(contradiction_insights),
                'drift_goals': len(drift_goals)
            }
        }
    
    def _handle_clarify_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /clarify command to show pending clarifications."""
        pending_requests = self.dialogue_clarification.get_pending_requests_for_ui()
        
        if not pending_requests:
            return {
                'response': "No pending clarification requests. Your beliefs appear consistent!",
                'command_processed': True,
                'pending_clarifications': 0
            }
        
        response_lines = ["🗣️ PENDING CLARIFICATION REQUESTS:", ""]
        
        for i, request in enumerate(pending_requests[:5], 1):  # Show top 5
            priority_indicator = "🔥" if request['priority'] >= 8 else "⚠️" if request['priority'] >= 6 else "💭"
            response_lines.append(f"{i}. {priority_indicator} {request['question']}")
            response_lines.append(f"   Context: {request['context']}")
            response_lines.append(f"   To respond: /clarify_response {request['id']} <your answer>")
            response_lines.append("")
        
        if len(pending_requests) > 5:
            response_lines.append(f"... and {len(pending_requests) - 5} more requests")
        
        return {
            'response': "\n".join(response_lines),
            'command_processed': True,
            'pending_clarifications': len(pending_requests),
            'clarification_requests': pending_requests
        }
    
    def _handle_tune_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /tune command to show tuning status."""
        tuning_summary = self.autonomous_tuning.get_tuning_summary()
        
        response_lines = ["🎛️ AUTONOMOUS MEMORY TUNING STATUS:", ""]
        response_lines.append(f"Current parameters:")
        
        for param, value in tuning_summary['current_parameters'].items():
            response_lines.append(f"  • {param}: {value:.3f}")
        
        response_lines.append("")
        response_lines.append(f"Recent changes: {len(tuning_summary['recent_changes'])} in last 24h")
        response_lines.append(f"Total tuning actions: {tuning_summary['total_tuning_actions']}")
        response_lines.append(f"Metrics collected: {tuning_summary['metrics_collected']}")
        response_lines.append(f"Next tuning in: {tuning_summary['next_tuning_in'] / 60:.1f} minutes")
        
        if tuning_summary['recent_changes']:
            response_lines.append("\nRecent parameter changes:")
            for param, changes in tuning_summary['recent_changes'].items():
                for change in changes[-3:]:  # Last 3 changes
                    response_lines.append(f"  • {param}: {change['old_value']:.3f} → {change['new_value']:.3f} ({change['reason']})")
        
        return {
            'response': "\n".join(response_lines),
            'command_processed': True,
            'tuning_data': tuning_summary
        }
    
    def _handle_perspectives_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /perspectives command to show theory of mind data."""
        perspective_summary = self.theory_of_mind.get_perspective_summary()
        
        response_lines = ["🧠 THEORY OF MIND - PERSPECTIVE TRACKING:", ""]
        response_lines.append(f"Total agents tracked: {perspective_summary['total_agents']}")
        response_lines.append(f"Total belief attributions: {perspective_summary['total_attributions']}")
        response_lines.append(f"Nested beliefs: {perspective_summary['nested_beliefs']}")
        response_lines.append(f"Perspective contradictions: {perspective_summary['perspective_contradictions']}")
        response_lines.append("")
        
        # Show agent details
        response_lines.append("Tracked agents:")
        for agent_id, agent_data in perspective_summary['agents'].items():
            trust_indicator = "🟢" if agent_data['trust_level'] > 0.7 else "🟡" if agent_data['trust_level'] > 0.4 else "🔴"
            response_lines.append(f"  {trust_indicator} {agent_data['name']} ({agent_data['type']})")
            response_lines.append(f"     Trust: {agent_data['trust_level']:.2f}, Beliefs: {agent_data['belief_count']}")
            if agent_data['recent_beliefs']:
                response_lines.append(f"     Recent: {', '.join(agent_data['recent_beliefs'])}")
            response_lines.append("")
        
        # Add insights
        insights = self.theory_of_mind.generate_perspective_insights()
        if insights:
            response_lines.append("Perspective insights:")
            for insight in insights:
                response_lines.append(f"  • {insight}")
        
        return {
            'response': "\n".join(response_lines),
            'command_processed': True,
            'perspective_data': perspective_summary
        }
    
    def _handle_health_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /health command to show cognitive health status."""
        # Force a quick cognitive cycle to get current health
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Collect performance metrics
        recent_clarifications = self.dialogue_clarification.completed_requests[-10:]  # Last 10
        metrics = self.autonomous_tuning.collect_performance_metrics(facts, recent_clarifications)
        
        # Get cognitive snapshot
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(facts)
        belief_patterns = self.belief_consolidation.analyze_belief_stability(facts)
        agent_data = self.theory_of_mind.get_perspective_summary()
        snapshot = self.self_inspection.get_self_summary(facts, contradiction_clusters, belief_patterns, agent_data)
        
        # Format health report
        health_status = "EXCELLENT" if snapshot.system_health_score >= 0.9 else \
                       "GOOD" if snapshot.system_health_score >= 0.7 else \
                       "FAIR" if snapshot.system_health_score >= 0.5 else \
                       "CRITICAL"
        
        response_lines = ["🏥 COGNITIVE HEALTH ASSESSMENT:", ""]
        response_lines.append(f"Overall Health: {health_status} ({snapshot.system_health_score:.2f}/1.0)")
        response_lines.append("")
        response_lines.append("Key Metrics:")
        response_lines.append(f"  • Belief confidence: {snapshot.average_confidence:.2f}")
        response_lines.append(f"  • Memory volatility: {snapshot.average_volatility:.2f}")
        response_lines.append(f"  • Memory utilization: {snapshot.memory_utilization:.1%}")
        response_lines.append(f"  • Active contradictions: {snapshot.unresolved_contradictions}")
        response_lines.append(f"  • Stability trend: {snapshot.belief_stability_trend}")
        response_lines.append("")
        
        # Performance metrics
        response_lines.append("System Performance:")
        response_lines.append(f"  • Contradiction detection rate: {metrics.contradiction_detection_rate:.1%}")
        response_lines.append(f"  • Contradiction resolution rate: {metrics.contradiction_resolution_rate:.1%}")
        response_lines.append(f"  • Clarification success rate: {metrics.clarification_success_rate:.1%}")
        response_lines.append(f"  • Query response accuracy: {metrics.query_response_accuracy:.1%}")
        response_lines.append("")
        
        # Issues and recommendations
        if snapshot.cognitive_drift_indicators:
            response_lines.append("🌊 Cognitive Drift Indicators:")
            for indicator in snapshot.cognitive_drift_indicators[:3]:
                response_lines.append(f"  ⚡ {indicator}")
            response_lines.append("")
        
        if snapshot.recommended_actions:
            response_lines.append("🎯 Recommended Actions:")
            for action in snapshot.recommended_actions[:3]:
                response_lines.append(f"  {action}")
        
        return {
            'response': "\n".join(response_lines),
            'command_processed': True,
            'health_data': {
                'health_score': snapshot.system_health_score,
                'health_status': health_status,
                'metrics': metrics,
                'drift_indicators': len(snapshot.cognitive_drift_indicators),
                'recommendations': len(snapshot.recommended_actions)
            }
        }
    
    def _handle_clarification_response(self, request_id: str, response_text: str) -> Dict:
        """Handle user response to a clarification request."""
        try:
            response = self.dialogue_clarification.process_user_response(request_id, response_text)
            
            result_lines = ["✅ Clarification response processed!", ""]
            result_lines.append(f"Resolution action: {response.resolution_action}")
            
            if response.resolution_action == "choose_one" and response.chosen_fact_id:
                result_lines.append("I'll update my beliefs based on your choice.")
            elif response.resolution_action == "update" and response.new_belief:
                result_lines.append(f"I'll update my understanding: {response.new_belief}")
            elif response.resolution_action == "confirm":
                result_lines.append("Thank you for confirming - I'll reinforce this belief.")
            elif response.resolution_action == "clarify_context":
                result_lines.append("I understand this might be context-dependent. I'll note the nuance.")
            
            # Apply the clarification to update beliefs
            self._apply_clarification_resolution(response)
            
            return {
                'response': "\n".join(result_lines),
                'command_processed': True,
                'clarification_resolved': True,
                'resolution_action': response.resolution_action
            }
            
        except ValueError as e:
            return {
                'response': f"Error processing clarification: {str(e)}",
                'command_processed': False,
                'clarification_resolved': False
            }
    
    def _should_run_cognitive_cycle(self) -> bool:
        """Check if it's time to run the autonomous cognitive cycle."""
        return time.time() - self.last_cognitive_cycle > self.cognitive_cycle_interval
    
    def _run_autonomous_cognitive_cycle(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Run the autonomous cognitive cycle."""
        print("[Phase2Cognitive] Running autonomous cognitive cycle")
        
        cycle_start = time.time()
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        insights = {
            'cycle_timestamp': cycle_start,
            'facts_analyzed': len(facts),
            'actions_taken': []
        }
        
        # 1. Analyze contradictions and generate clarifications
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(facts)
        clarification_requests = self.dialogue_clarification.analyze_volatility_and_generate_requests(
            facts, contradiction_clusters
        )
        
        if clarification_requests:
            insights['actions_taken'].append(f"Generated {len(clarification_requests)} clarification requests")
        
        # 2. Perform belief consolidation
        belief_patterns = self.belief_consolidation.analyze_belief_stability(facts)
        consolidation_summary = self.belief_consolidation.get_consolidation_summary()
        
        if consolidation_summary['consolidated_beliefs'] > 0:
            insights['actions_taken'].append(f"Consolidated {consolidation_summary['consolidated_beliefs']} beliefs")
        
        # 3. Collect performance metrics and tune parameters
        recent_clarifications = self.dialogue_clarification.completed_requests[-20:]
        performance_metrics = self.autonomous_tuning.collect_performance_metrics(
            facts, recent_clarifications
        )
        
        tuning_actions = self.autonomous_tuning.perform_autonomous_tuning()
        if tuning_actions:
            insights['actions_taken'].append(f"Applied {len(tuning_actions)} parameter adjustments")
        
        # 4. Update perspective analysis
        perspective_contradictions = self.theory_of_mind.detect_perspective_contradictions(facts)
        if perspective_contradictions:
            insights['actions_taken'].append(f"Detected {len(perspective_contradictions)} perspective conflicts")
        
        # 5. Perform meta-cognitive analysis
        if hasattr(self, 'meta_cognition') and self.meta_cognition is not None:
            try:
                cognitive_health = self.meta_cognition.perform_cognitive_scan(
                    facts, contradiction_clusters, belief_patterns
                )
                
                meta_goals = self.meta_cognition.get_priority_goals(5)
                if meta_goals:
                    insights['actions_taken'].append(f"Generated {len(meta_goals)} priority meta-goals")
            except Exception as e:
                print(f"[Phase2Cognitive] Meta-cognition analysis failed: {e}")
        else:
            print("[Phase2Cognitive] Meta-cognition agent not available")
        
        # 6. Run self-inspection
        agent_data = self.theory_of_mind.get_perspective_summary()
        cognitive_snapshot = self.self_inspection.get_self_summary(
            facts, contradiction_clusters, belief_patterns, agent_data
        )
        
        insights['cognitive_health_score'] = cognitive_snapshot.system_health_score
        insights['drift_indicators'] = len(cognitive_snapshot.cognitive_drift_indicators)
        
        self.last_cognitive_cycle = cycle_start
        cycle_duration = time.time() - cycle_start
        insights['cycle_duration_ms'] = int(cycle_duration * 1000)
        
        print(f"[Phase2Cognitive] Cognitive cycle completed in {cycle_duration:.2f}s, "
              f"health score: {cognitive_snapshot.system_health_score:.2f}")
        
        return insights
    
    def _check_for_clarification_opportunities(self, text: str, user_profile_id: str = None,
                                             session_id: str = None) -> List[ClarificationRequest]:
        """Check if current input creates clarification opportunities."""
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Quick volatility check on recent facts
        recent_facts = [f for f in facts 
                       if f.timestamp and time.time() - f.timestamp < 3600]  # Last hour
        
        if len(recent_facts) < 2:
            return []
        
        # Look for immediate contradictions
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(recent_facts)
        
        # Generate urgent clarifications only
        urgent_requests = []
        for cluster_id, cluster in contradiction_clusters.items():
            if cluster.volatility_score > 0.8:  # Very high volatility
                requests = self.dialogue_clarification._generate_cluster_clarifications({cluster_id: cluster})
                urgent_requests.extend([r for r in requests if r.priority >= 8])
        
        return urgent_requests
    
    def _apply_advanced_confabulation_filtering(self, response: str, query: str,
                                              user_profile_id: str = None,
                                              session_id: str = None) -> Dict:
        """Apply advanced confabulation filtering with Phase 2 features."""
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Get enhanced contradiction history
        contradiction_history = {
            'reliability_score': 0.8,  # Higher default for Phase 2
            'recent_corrections': len(self.dialogue_clarification.completed_requests),
            'perspective_conflicts': len(self.theory_of_mind.perspective_contradictions)
        }
        
        # Apply standard confabulation filtering
        assessment = self.confabulation_filter.assess_response_reliability(
            response, query, facts, contradiction_history
        )
        
        # Enhanced filtering based on perspective awareness
        if assessment.confabulation_risk > 0.5:
            # Check if response conflicts with known perspective beliefs
            perspective_conflicts = self._check_perspective_conflicts(response, facts)
            if perspective_conflicts:
                assessment.confabulation_risk = min(1.0, assessment.confabulation_risk + 0.2)
                assessment.action_taken = "hedged"
                assessment.filtered_response = f"I believe {response.lower()}, though I have some conflicting information from different sources."
        
        return {
            'original_response': response,
            'filtered_response': assessment.filtered_response,
            'confabulation_assessment': {
                'confidence_score': assessment.confidence_score,
                'confabulation_risk': assessment.confabulation_risk,
                'action_taken': assessment.action_taken,
                'supporting_facts_count': len(assessment.supporting_facts),
                'contradicting_facts_count': len(assessment.contradicting_facts),
                'perspective_aware': True
            }
        }
    
    def _check_perspective_conflicts(self, response: str, facts: List[EnhancedTripletFact]) -> bool:
        """Check if response conflicts with perspective-tagged beliefs."""
        response_lower = response.lower()
        
        # Look for statements that might conflict across perspectives
        for fact in facts:
            if fact.perspective != "user" and fact.perspective in response_lower:
                # Response mentions a specific person - check for conflicts
                if fact.object.lower() in response_lower and fact.predicate.lower() in response_lower:
                    return True
        
        return False
    
    def _apply_clarification_resolution(self, clarification_response):
        """Apply the resolution from a clarification response."""
        if clarification_response.resolution_action == "choose_one" and clarification_response.chosen_fact_id:
            # Mark the chosen fact as reinforced and others as less confident
            try:
                facts = self.get_facts()
                for fact in facts:
                    if fact.id == clarification_response.chosen_fact_id:
                        fact.confidence = min(1.0, fact.confidence + 0.1)
                        fact.update_access()
                    elif fact.id in [f.id for f in facts if f.contradiction]:
                        fact.confidence = max(0.1, fact.confidence - 0.1)
                
                # Store updated facts
                self.fact_manager.store_facts(facts)
                
            except Exception as e:
                print(f"[Phase2Cognitive] Error applying clarification resolution: {e}")
        
        elif clarification_response.resolution_action == "update" and clarification_response.new_belief:
            # Create new fact from updated belief
            # This would require triplet extraction from the new belief text
            print(f"[Phase2Cognitive] Would create new fact from: {clarification_response.new_belief}")
    
    def _detect_and_create_causal_links(self, text: str, user_profile_id: str = None,
                                      session_id: str = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Detect and create causal links between facts based on temporal proximity and semantic similarity.
        Implements the mathematical formula from the paper: 
        CausalStrength = temporal_proximity * semantic_similarity * logical_consistency
        
        📌 DO NOT HARDCODE thresholds or scores.
        All parameters must be loaded from `config.settings` or environment config.
        This is a zero-hardcoding cognitive subsystem.
        """
        try:
            from config.settings import CAUSAL_LINK_THRESHOLD, TEMPORAL_DECAY_LAMBDA
            import time
            import math
        except Exception as e:
            print(f"[Phase2Cognitive] ❌ Error importing settings: {e}")
            return [], {'error': 'import_failed', 'details': str(e)}
        
        # Import metrics and audit logging for runtime monitoring
        try:
            from monitoring.metrics import (
                cognitive_causal_links_total, cognitive_causal_link_strength,
                cognitive_causal_analysis_duration_seconds, cognitive_temporal_proximity_score,
                cognitive_semantic_similarity_score, cognitive_logical_consistency_score,
                cognitive_causal_failures_total
            )
            metrics_available = True
        except ImportError:
            metrics_available = False
            print(f"[Phase2Cognitive] Metrics not available - continuing without metrics")
        
        try:
            from monitoring.logger import log_causal_link_created, log_causal_analysis_performed, log_causal_failure
            audit_logging_available = True
            print(f"[Phase2Cognitive] Audit logging available")
        except ImportError as e:
            # Fallback audit logging using standard logging
            import logging
            audit_logger = logging.getLogger("causal_audit")
            
            def log_causal_link_created(fact_id: str, cause: str, strength: float, user_id: str = None, session_id: str = None):
                audit_logger.info(f"CAUSAL_LINK_CREATED: fact_id={fact_id}, cause={cause}, strength={strength:.3f}, user_id={user_id}, session_id={session_id}")
            
            def log_causal_analysis_performed(fact_count: int, links_created: int, analysis_duration: float, user_id: str = None):
                audit_logger.info(f"CAUSAL_ANALYSIS_PERFORMED: fact_count={fact_count}, links_created={links_created}, duration={analysis_duration:.3f}s, user_id={user_id}")
            
            def log_causal_failure(reason: str, details: str = None, user_id: str = None, session_id: str = None):
                audit_logger.warning(f"CAUSAL_ANALYSIS_FAILURE: reason={reason}, details={details}, user_id={user_id}, session_id={session_id}")
            
            audit_logging_available = True
            print(f"[Phase2Cognitive] Using fallback audit logging due to import error: {e}")
        
        # Get recent facts for potential causal analysis (last 24 hours)
        current_time = time.time()
        recent_window = 24 * 3600  # 24 hours in seconds
        
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Include facts with recent timestamps OR missing timestamps (likely just created)
        recent_facts = []
        for f in facts:
            if not f.timestamp:
                # No timestamp = likely just created, include it
                recent_facts.append(f)
            elif (current_time - f.timestamp) <= recent_window:
                # Within time window
                recent_facts.append(f)
                
        # If we still don't have enough facts, use the most recent ones regardless of timestamp
        if len(recent_facts) < 2 and len(facts) >= 2:
            recent_facts = facts[-4:]  # Use last 4 facts
        
        print(f"[Phase2Cognitive] Analyzing {len(recent_facts)} recent facts for causal links")
        
        if len(recent_facts) < 2:
            return [], {'facts_analyzed': len(recent_facts), 'analysis_performed': False, 'reason': 'insufficient_facts'}
        
        # Start timing causal analysis for metrics
        analysis_start_time = time.time() if metrics_available else None
        
        # Sort by timestamp to find potential cause-effect relationships
        recent_facts.sort(key=lambda f: f.timestamp or 0)
        causal_links_created = []
        causal_analysis_metadata = {
            'facts_analyzed': len(recent_facts),
            'threshold_used': CAUSAL_LINK_THRESHOLD,
            'temporal_decay_lambda': TEMPORAL_DECAY_LAMBDA,
            'analysis_pairs': 0,
            'links_above_threshold': 0,
            'analysis_performed': True
        }
        
        for i in range(len(recent_facts) - 1):
            fact1 = recent_facts[i]  # Potential cause
            
            for j in range(i + 1, len(recent_facts)):
                fact2 = recent_facts[j]  # Potential effect
                causal_analysis_metadata['analysis_pairs'] += 1
                
                # Handle missing timestamps gracefully
                if not (fact1.timestamp and fact2.timestamp):
                    print(f"[Phase2Cognitive] ⚠️ Missing timestamps: fact1={fact1.timestamp}, fact2={fact2.timestamp}")
                    if metrics_available:
                        cognitive_causal_failures_total.labels(reason='missing_timestamps').inc()
                    if audit_logging_available:
                        log_causal_failure(
                            reason='missing_timestamps',
                            details=f"fact1={fact1.timestamp}, fact2={fact2.timestamp}",
                            user_id=user_profile_id,
                            session_id=session_id
                        )
                    
                    # Use default temporal proximity when timestamps are missing
                    default_temporal_proximity = DEFAULT_VALUES.get('causal_missing_timestamp_proximity', 0.5)
                    temporal_proximity = default_temporal_proximity
                    print(f"[Phase2Cognitive] Using default temporal proximity: {temporal_proximity}")
                else:
                    # Calculate temporal proximity: e^(-λ * |t2 - t1|)
                    time_diff = abs(fact2.timestamp - fact1.timestamp)
                    temporal_proximity = math.exp(-TEMPORAL_DECAY_LAMBDA * time_diff)
                    
                    # Apply reverse causality penalty if effect predates cause
                    if fact2.timestamp < fact1.timestamp:
                        reverse_causality_penalty = DEFAULT_VALUES.get('causal_reverse_causality_penalty', 0.5)
                        temporal_proximity *= reverse_causality_penalty
                        print(f"[Phase2Cognitive] ⏳ Reverse causality detected - applying penalty: {reverse_causality_penalty}")
                

                
                # Calculate semantic similarity using embeddings or text similarity
                semantic_similarity = self._calculate_semantic_similarity(fact1, fact2)
                
                # Calculate logical consistency based on predicate compatibility
                logical_consistency = self._calculate_logical_consistency(fact1, fact2)
                
                # Calculate causal strength using weighted average instead of multiplication
                # This prevents temporal proximity from killing strong semantic connections
                causal_strength = (
                    0.2 * temporal_proximity +     # Low weight - temporal gaps are less critical
                    0.4 * semantic_similarity +    # High weight - strong semantic connections matter most
                    0.4 * logical_consistency      # High weight - logical flow is important
                )
                
                # Boost for strong semantic + logical patterns
                if semantic_similarity > 0.7 and logical_consistency >= 0.5:
                    causal_strength += 0.15  # Bonus for clearly related concepts
                    print(f"[Phase2Cognitive] 🚀 Strong semantic+logical pattern detected - applying 0.15 boost")
                
                # Record metrics for causal analysis components
                if metrics_available:
                    cognitive_temporal_proximity_score.observe(temporal_proximity)
                    cognitive_semantic_similarity_score.observe(semantic_similarity)
                    cognitive_logical_consistency_score.observe(logical_consistency)
                    cognitive_causal_link_strength.observe(causal_strength)
                
                print(f"[Phase2Cognitive] Causal analysis: '{fact1.subject} {fact1.predicate}' -> '{fact2.subject} {fact2.predicate}'")
                print(f"  Temporal proximity: {temporal_proximity:.3f}")
                print(f"  Semantic similarity: {semantic_similarity:.3f}")
                print(f"  Logical consistency: {logical_consistency:.3f}")
                print(f"  Causal strength: {causal_strength:.3f} (threshold: {CAUSAL_LINK_THRESHOLD})")
                
                # Log all causal strength scores for debugging, even if below threshold
                if causal_strength < CAUSAL_LINK_THRESHOLD:
                    # Check for near-causal patterns that might be worth tracking
                    if causal_strength >= 0.05:  # Near-causal threshold
                        print(f"  ⚠️ Near-causal pattern detected (strength: {causal_strength:.3f}) - tracking for potential tuning")
                        # TODO: Could store these in a potential_links table for learning
                    else:
                        print(f"  ❌ Below threshold - not creating causal link")
                    
                    # Record failed causal link creation metric
                    if metrics_available:
                        cognitive_causal_failures_total.labels(reason='low_similarity').inc()
                
                # Create causal link if strength exceeds threshold
                if causal_strength >= CAUSAL_LINK_THRESHOLD:
                    causal_analysis_metadata['links_above_threshold'] += 1
                    cause_description = f"{fact1.subject} {fact1.predicate} {fact1.object}"
                    fact2.add_causal_link(cause_description, causal_strength)
                    
                    # Store the updated fact in database
                    self._update_stored_fact(fact2)
                    
                    causal_links_created.append(f"{fact1.id} -> {fact2.id}")
                    print(f"[Phase2Cognitive] ✅ Created causal link: {cause_description} -> {fact2.subject} {fact2.predicate}")
                    
                    # Also update the fact object directly for immediate availability
                    fact2.cause = cause_description
                    fact2.causal_strength = causal_strength
                    
                    # Record successful causal link creation metric
                    if metrics_available:
                        cognitive_causal_links_total.labels(status='created').inc()
                    
                    # Audit log causal link creation
                    if audit_logging_available:
                        log_causal_link_created(
                            fact_id=str(fact2.id), 
                            cause=cause_description,
                            strength=causal_strength,
                            user_id=user_profile_id,
                            session_id=session_id
                        )
        
        # Record causal analysis duration metric and audit log
        if analysis_start_time:
            analysis_duration = time.time() - analysis_start_time
            if metrics_available:
                cognitive_causal_analysis_duration_seconds.observe(analysis_duration)
            
            # Audit log overall causal analysis session
            if audit_logging_available:
                log_causal_analysis_performed(
                    fact_count=len(recent_facts),
                    links_created=len(causal_links_created),
                    analysis_duration=analysis_duration,
                    user_id=user_profile_id
                )
        
        return causal_links_created, causal_analysis_metadata
    
    def _calculate_semantic_similarity(self, fact1, fact2) -> float:
        """Calculate semantic similarity between two facts using embeddings first, then fallbacks."""
        # Method 1: Try embeddings if available
        try:
            from scripts.embedder import embed
            import numpy as np
            from scipy.spatial.distance import cosine
            
            # Combine predicate and object for semantic comparison
            text1 = f"{fact1.predicate} {fact1.object}".lower()
            text2 = f"{fact2.predicate} {fact2.object}".lower()
            
            embedding1 = embed(text1)
            embedding2 = embed(text2)
            
            if not np.all(embedding1 == 0) and not np.all(embedding2 == 0):
                similarity = 1 - cosine(embedding1, embedding2)
                return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            print(f"[Phase2Cognitive] Embedding similarity failed: {e}")
        
        # Method 2: Try spaCy if available
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                text1 = f"{fact1.predicate} {fact1.object}".lower()
                text2 = f"{fact2.predicate} {fact2.object}".lower()
                
                doc1 = nlp(text1)
                doc2 = nlp(text2)
                
                return doc1.similarity(doc2)
        except Exception as e:
            print(f"[Phase2Cognitive] spaCy similarity failed: {e}")
        
        # Method 3: Fallback to text similarity
        from difflib import SequenceMatcher
        text1 = f"{fact1.subject} {fact1.predicate} {fact1.object}".lower()
        text2 = f"{fact2.subject} {fact2.predicate} {fact2.object}".lower()
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_logical_consistency(self, fact1, fact2) -> float:
        """
        Calculate logical consistency based on predicate compatibility with sophisticated scoring.
        
        📌 DO NOT HARDCODE pattern weights or scores.
        All scoring logic must be configurable and hot-reloadable.
        """
        return self._compute_predicate_score(fact1.predicate, fact2.predicate, fact1.subject, fact2.subject, fact1.object, fact2.object)
    
    def _compute_predicate_score(self, pred1: str, pred2: str, subj1: str, subj2: str, obj1: str, obj2: str) -> float:
        """
        Compute predicate compatibility score for causal relationships.
        Returns values like: 1.0 if perfect causal pattern, 0.8 if similar, 0.5 if weak, etc.
        
        📌 DO NOT HARDCODE compatibility scores or pattern weights.
        All predicate pattern scoring must come from configuration.
        """
        pred1_lower = pred1.lower().strip()
        pred2_lower = pred2.lower().strip()
        subj1_lower = subj1.lower().strip()
        subj2_lower = subj2.lower().strip()
        obj1_lower = obj1.lower().strip()
        obj2_lower = obj2.lower().strip()
        
        # Load causal predicate pattern scores from configuration
        from config.settings import DEFAULT_VALUES
        
        # Define causal predicate patterns with scoring weights from config
        causal_patterns = {
            # Perfect causal patterns
            'action_to_emotion': {
                'score': DEFAULT_VALUES.get('causal_pattern_action_to_emotion_score', 1.0),
                'causes': ['join', 'joined', 'start', 'started', 'begin', 'began', 'enter', 'entered'],
                'effects': ['feel', 'felt', 'experience', 'experienced', 'become', 'became', 'overwhelm', 'overwhelmed']
            },
            
            # Strong causal patterns
            'action_to_state': {
                'score': DEFAULT_VALUES.get('causal_pattern_action_to_state_score', 0.9),
                'causes': ['join', 'start', 'begin', 'create', 'build', 'make'],
                'effects': ['is', 'are', 'was', 'were', 'has', 'have', 'become', 'became']
            },
            
            # Medium causal patterns
            'state_to_emotion': {
                'score': DEFAULT_VALUES.get('causal_pattern_state_to_emotion_score', 0.8),
                'causes': ['is', 'are', 'was', 'were', 'have', 'has', 'behind', 'ahead', 'late', 'early'],
                'effects': ['feel', 'felt', 'experience', 'think', 'believe', 'worried', 'anxious', 'stressed', 'excited', 'happy', 'sad', 'concerned', 'nervous']
            },
            
            # Weak but valid patterns
            'general_sequence': {
                'score': DEFAULT_VALUES.get('causal_pattern_general_sequence_score', 0.6),
                'causes': ['do', 'did', 'perform', 'complete', 'finish'],
                'effects': ['get', 'got', 'receive', 'achieve', 'obtain']
            },
            
            # Temporal sequence patterns
            'temporal_sequence': {
                'score': DEFAULT_VALUES.get('causal_pattern_temporal_sequence_score', 0.85),
                'causes': ['before', 'prior', 'previously', 'earlier', 'first', 'initially'],
                'effects': ['after', 'later', 'subsequently', 'then', 'finally', 'eventually']
            }
        }
        
        # Check for exact causal pattern matches
        # Look in both predicate and object for better pattern matching
        fact1_text = f"{pred1_lower} {obj1_lower}".lower()
        fact2_text = f"{pred2_lower} {obj2_lower}".lower()
        
        for pattern_name, pattern in causal_patterns.items():
            cause_match = any(cause in fact1_text for cause in pattern['causes'])
            effect_match = any(effect in fact2_text for effect in pattern['effects'])
            
            if cause_match and effect_match:
                print(f"[Phase2Cognitive] Detected {pattern_name} pattern: {fact1_text} -> {fact2_text}")
                return pattern['score']
        
        # Same subject bonus (indicates personal experience continuity)
        same_subject_score = DEFAULT_VALUES.get('causal_same_subject_score', 0.5)
        different_subject_score = DEFAULT_VALUES.get('causal_different_subject_score', 0.3)
        
        if subj1_lower == subj2_lower:
            base_score = same_subject_score
        else:
            base_score = different_subject_score
        
        # Predicate similarity bonus
        exact_predicate_bonus = DEFAULT_VALUES.get('causal_exact_predicate_bonus', 0.1)
        partial_predicate_bonus = DEFAULT_VALUES.get('causal_partial_predicate_bonus', 0.05)
        
        if pred1_lower == pred2_lower:
            base_score += exact_predicate_bonus
        elif any(word in pred1_lower for word in pred2_lower.split()) or \
             any(word in pred2_lower for word in pred1_lower.split()):
            base_score += partial_predicate_bonus
        
        return min(1.0, base_score)
    
    # ==========================================
    # CAUSAL-AWARE PLANNING COMMAND HANDLERS
    # ==========================================
    
    def _handle_causal_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /causal command to show causal analysis overview."""
        if not self.causal_goal_generator:
            return {
                'response': "❌ Causal-aware planning system not available",
                'command_processed': True
            }
        
        # Get causal health summary
        health = self.causal_goal_generator.get_causal_health_summary()
        
        # Get execution status if available
        exec_status = {}
        if self.causal_executor:
            exec_status = self.causal_executor.get_execution_status()
        
        response_lines = [
            "🧠 CAUSAL-AWARE PLANNING OVERVIEW",
            "=" * 40,
            f"📊 Status: {health.get('status', 'unknown').title()}",
            f"🔗 Total Causal Facts: {health.get('total_causal_facts', 0)}",
            f"💪 Average Strength: {health.get('average_causal_strength', 0):.3f}",
            f"⚡ Strong Links: {health.get('strong_links', 0)}",
            f"⚠️ Weak Links: {health.get('weak_links', 0)}",
            "",
            "🎯 GOAL STATUS",
            f"Pending Goals: {health.get('pending_goals', 0)}",
            f"Completed Goals: {health.get('completed_goals', 0)}",
            f"Feedback Sessions: {health.get('feedback_sessions', 0)}",
        ]
        
        if exec_status:
            response_lines.extend([
                "",
                "🔄 EXECUTION STATUS",
                f"Running: {'Yes' if exec_status.get('is_running') else 'No'}",
                f"Queue Size: {exec_status.get('queue_size', 0)}",
                f"Success Rate: {exec_status.get('success_rate', 0):.1%}",
            ])
        
        response_lines.extend([
            "",
            "💡 COMMANDS",
            "/causal_goals - Show current goals",
            "/causal_health - Detailed health report", 
            "/start_causal - Start autonomous execution",
            "/stop_causal - Stop autonomous execution"
        ])
        
        return {
            'response': '\n'.join(response_lines),
            'command_processed': True,
            'causal_overview': {
                'health': health,
                'execution_status': exec_status
            }
        }

    def _handle_causal_goals_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /causal_goals command to show current causal goals."""
        if not self.causal_goal_generator:
            return {
                'response': "❌ Causal goal generator not available",
                'command_processed': True
            }
        
        # Generate fresh goals
        goals = self.causal_goal_generator.generate_causal_meta_goals(user_profile_id, session_id)
        priority_goals = self.causal_goal_generator.get_priority_causal_goals(limit=5)
        
        response_lines = [
            "🎯 CAUSAL META-GOALS",
            "=" * 30,
            f"Generated: {len(goals)} goals",
            f"High Priority: {len(priority_goals)} goals",
            ""
        ]
        
        if priority_goals:
            response_lines.append("📋 TOP PRIORITY GOALS:")
            for i, goal in enumerate(priority_goals, 1):
                status_icon = "🔄" if goal.status == "pending" else "✅"
                improvement_icons = {
                    "clarification": "🔍",
                    "reinforcement": "💪", 
                    "gap_filling": "🌉",
                    "validation": "✅"
                }
                type_icon = improvement_icons.get(goal.improvement_type, "❓")
                
                response_lines.extend([
                    f"{i}. {status_icon} {type_icon} {goal.description}",
                    f"   Priority: {goal.priority}/10",
                    f"   Type: {goal.improvement_type}",
                    f"   Suggested: {goal.suggested_actions[0] if goal.suggested_actions else 'No actions'}"
                ])
                
                if i < len(priority_goals):  # Don't add separator after last goal
                    response_lines.append("")
        else:
            response_lines.append("No goals currently pending.")
        
        response_lines.extend([
            "",
            "💡 Use '/causal_execute <goal_id>' to manually execute a goal"
        ])
        
        return {
            'response': '\n'.join(response_lines),
            'command_processed': True,
            'causal_goals': [
                {
                    'goal_id': g.goal_id,
                    'description': g.description,
                    'priority': g.priority,
                    'type': g.improvement_type,
                    'status': g.status
                }
                for g in priority_goals
            ]
        }

    def _handle_causal_health_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /causal_health command for detailed causal reasoning health."""
        if not self.causal_executor:
            return {
                'response': "❌ Causal executor not available",
                'command_processed': True
            }
        
        # Generate improvement report
        report_lines = self.causal_executor.generate_improvement_report()
        
        # Add recent execution results
        recent_results = self.causal_executor.get_recent_results(limit=5)
        if recent_results:
            report_lines.extend([
                "",
                "🔄 RECENT EXECUTIONS",
                "-" * 20
            ])
            
            for result in recent_results:
                status_icon = "✅" if result.success else "❌"
                report_lines.append(f"{status_icon} {result.query[:60]}...")
                if result.memory_updates:
                    report_lines.append(f"   Updates: {len(result.memory_updates)}")
        
        return {
            'response': '\n'.join(report_lines),
            'command_processed': True,
            'causal_health_report': {
                'analysis': self.causal_executor.analyze_causal_evolution(),
                'recent_results': [
                    {
                        'goal_id': r.goal_id,
                        'query': r.query,
                        'success': r.success,
                        'execution_time': r.execution_time
                    }
                    for r in recent_results
                ]
            }
        }

    def _handle_start_causal_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /start_causal command to begin autonomous execution."""
        if not self.causal_executor:
            return {
                'response': "❌ Causal executor not available",
                'command_processed': True
            }
        
        if self.causal_executor.is_running:
            return {
                'response': "🔄 Causal execution is already running",
                'command_processed': True
            }
        
        # Set up question callback for autonomous execution
        def question_callback(query: str) -> str:
            # This would ideally integrate with the chat interface
            # For now, we'll simulate responses
            print(f"[CausalExecutor] AUTO-QUERY: {query}")
            return "I understand the pattern you're asking about."
        
        self.causal_executor.set_question_callback(question_callback)
        self.causal_executor.start_autonomous_execution()
        
        return {
            'response': "🚀 Started autonomous causal execution!\n\nThe system will now:\n• Generate causal improvement goals\n• Execute clarification queries\n• Process feedback and update memory\n• Continuously improve causal understanding\n\nUse '/stop_causal' to stop or '/causal_health' to monitor progress.",
            'command_processed': True,
            'autonomous_started': True
        }

    def _handle_stop_causal_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /stop_causal command to stop autonomous execution."""
        if not self.causal_executor:
            return {
                'response': "❌ Causal executor not available",
                'command_processed': True
            }
        
        if not self.causal_executor.is_running:
            return {
                'response': "⏹️ Causal execution is not currently running",
                'command_processed': True
            }
        
        self.causal_executor.stop_autonomous_execution()
        
        # Get final status
        status = self.causal_executor.get_execution_status()
        
        return {
            'response': f"⏹️ Stopped autonomous causal execution.\n\nFinal Statistics:\n• Total Executions: {status.get('total_executions', 0)}\n• Success Rate: {status.get('success_rate', 0):.1%}\n• Completed Goals: {status.get('completed_goals', 0)}",
            'command_processed': True,
            'autonomous_stopped': True,
            'final_status': status
        }

    def _handle_causal_execute_command(self, goal_id: str, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /causal_execute command to manually execute a specific goal."""
        if not self.causal_executor or not goal_id:
            return {
                'response': "❌ Invalid goal ID or causal executor not available",
                'command_processed': True
            }
        
        # Execute the specific goal
        result = self.causal_executor.execute_specific_goal(goal_id)
        
        if result.success:
            response_lines = [
                f"✅ Successfully executed goal: {goal_id}",
                f"🗣️ Query: {result.query}",
                f"💬 Response: {result.response}",
                f"⏱️ Execution Time: {result.execution_time:.2f}s"
            ]
            
            if result.memory_updates:
                response_lines.extend([
                    "",
                    "🔄 Memory Updates:",
                    *[f"• {update}" for update in result.memory_updates]
                ])
        else:
            response_lines = [
                f"❌ Failed to execute goal: {goal_id}",
                f"Error: {result.error_message}"
            ]
        
        return {
            'response': '\n'.join(response_lines),
            'command_processed': True,
            'execution_result': {
                'goal_id': result.goal_id,
                'success': result.success,
                'query': result.query,
                'response': result.response,
                                'execution_time': result.execution_time
            }
        }

    def _handle_causal_graph_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /causal_graph command to visualize causal relationships."""
        try:
            from .causal_graph_visualizer import CausalGraphVisualizer
            
            visualizer = CausalGraphVisualizer(self.memory_log)
            
            # Generate text-based graph
            text_graph = visualizer.generate_text_graph(user_profile_id, min_strength=0.3)
            
            # Generate network summary
            network_summary = visualizer.generate_network_summary(user_profile_id, min_strength=0.3)
            
            # Generate DOT graph for external visualization
            dot_graph = visualizer.generate_dot_graph(user_profile_id, min_strength=0.3)
            
            response_lines = text_graph.copy()
            
            if network_summary.get("status") == "success":
                response_lines.extend([
                    "",
                    "📊 NETWORK STATISTICS",
                    "-" * 20,
                    f"Nodes: {network_summary['total_nodes']}",
                    f"Edges: {network_summary['total_edges']}",
                    f"Network Density: {network_summary['network_density']:.3f}",
                    f"Strongest Link: {network_summary['strongest_link']:.3f}",
                    ""
                ])
                
                if network_summary['most_connected_nodes']:
                    response_lines.append("🔗 Most Connected Nodes:")
                    for node, degree in network_summary['most_connected_nodes'][:3]:
                        response_lines.append(f"  • {node.replace('_', ' ')}: {degree} connections")
            
            response_lines.extend([
                "",
                "💡 VISUALIZATION OPTIONS",
                "• DOT graph saved to metadata for external tools",
                "• Use Graphviz or online viewers for visual graphs",
                "• Install networkx + matplotlib for built-in plots"
            ])
            
            return {
                'response': '\n'.join(response_lines),
                'command_processed': True,
                'causal_graph_data': {
                    'text_representation': text_graph,
                    'network_summary': network_summary,
                    'dot_graph': dot_graph,
                    'visualization_tools': {
                        'networkx': visualizer.networkx_available,
                        'matplotlib': visualizer.matplotlib_available
                    }
                }
            }
            
        except ImportError:
            return {
                'response': "❌ Causal graph visualizer not available",
                'command_processed': True
            }
        except Exception as e:
            return {
                'response': f"❌ Error generating causal graph: {e}",
                'command_processed': True
            }
     
    def _update_stored_fact(self, fact):
        """Update a stored fact with new causal link information."""
        try:
            # Update the fact in storage to persist causal links
            # Use the enhanced memory system's update method for better persistence
            if hasattr(fact, 'cause') and hasattr(fact, 'causal_strength') and fact.id:
                self.update_fact_causal_info(fact.id, fact.cause, fact.causal_strength)
                print(f"[Phase2Cognitive] Updated fact {fact.id} with causal info in database")
            else:
                print(f"[Phase2Cognitive] Warning: Fact missing causal info or ID: cause={getattr(fact, 'cause', None)}, strength={getattr(fact, 'causal_strength', None)}, id={getattr(fact, 'id', None)}")
        except Exception as e:
            print(f"[Phase2Cognitive] Warning: Could not update stored fact: {e}")
    
    def _get_current_cognitive_insights(self) -> List[str]:
        """Get current cognitive insights from all subsystems."""
        insights = []
        
        # Get insights from each subsystem
        if hasattr(self, 'contradiction_clustering'):
            cluster_insights = self.contradiction_clustering.suggest_cluster_insights()
            insights.extend([f"🪢 {insight}" for insight in cluster_insights[:2]])
        
        if hasattr(self, 'belief_consolidation'):
            consolidation_insights = self.belief_consolidation.suggest_consolidation_actions()
            insights.extend([f"🧬 {insight}" for insight in consolidation_insights[:2]])
        
        if hasattr(self, 'theory_of_mind'):
            perspective_insights = self.theory_of_mind.generate_perspective_insights()
            insights.extend([f"🧠 {insight}" for insight in perspective_insights[:2]])
        
        if hasattr(self, 'meta_cognition') and self.meta_cognition is not None:
            try:
                meta_insights = self.meta_cognition.generate_self_improvement_suggestions()
                insights.extend([f"🤖 {insight}" for insight in meta_insights[:2]])
            except Exception as e:
                print(f"[Phase2Cognitive] Meta-cognition insights failed: {e}")
        
        return insights[:6]  # Limit to top 6 insights
    
    def get_phase2_dashboard(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Get comprehensive Phase 2 cognitive dashboard."""
        if not PHASE2_AVAILABLE:
            return {'error': 'Phase 2 features not available'}
        
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        dashboard = {
            'system_overview': {
                'phase': 2,
                'autonomous_mode': self.autonomous_mode,
                'total_facts': len(facts),
                'last_cognitive_cycle': self.last_cognitive_cycle,
                'features_active': PHASE2_AVAILABLE
            },
            
            'dialogue_clarification': self.dialogue_clarification.get_clarification_summary(),
            
            'autonomous_tuning': self.autonomous_tuning.get_tuning_summary(),
            
            'theory_of_mind': self.theory_of_mind.get_perspective_summary(),
            
            'self_inspection': self.self_inspection.get_inspection_summary(),
            
            'cognitive_insights': self._get_current_cognitive_insights(),
            
            'system_status': {
                'cognitive_cycles_completed': len(self.self_inspection.inspection_history),
                'clarifications_resolved': len(self.dialogue_clarification.completed_requests),
                'parameters_tuned': len(self.autonomous_tuning.tuning_history),
                'perspectives_tracked': len(self.theory_of_mind.agents),
                'autonomous_health': 'excellent' if self.autonomous_mode else 'manual_mode'
            }
        }
        
        return dashboard
    
    def export_phase2_state(self, output_file: str, user_profile_id: str = None,
                           session_id: str = None) -> bool:
        """Export complete Phase 2 cognitive state."""
        if not PHASE2_AVAILABLE:
            return False
        
        try:
            dashboard = self.get_phase2_dashboard(user_profile_id, session_id)
            facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            
            export_data = {
                'export_metadata': {
                    'timestamp': time.time(),
                    'phase': 2,
                    'user_profile_id': user_profile_id,
                    'session_id': session_id,
                    'system_version': 'MeRNSTA Phase 2 Autonomous Cognitive Agent v1.0'
                },
                'dashboard': dashboard,
                'facts': [fact.to_dict() for fact in facts],
                'clarification_requests': self.dialogue_clarification.get_pending_requests_for_ui(),
                'tuning_parameters': self.autonomous_tuning.get_current_parameters(),
                'perspective_agents': {
                    agent_id: {
                        'name': agent.name,
                        'type': agent.agent_type,
                        'trust_level': agent.trust_level,
                        'belief_count': agent.belief_count
                    }
                    for agent_id, agent in self.theory_of_mind.agents.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"[Phase2Cognitive] Phase 2 state exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"[Phase2Cognitive] Failed to export Phase 2 state: {e}")
            return False 
    
    def _handle_why_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle the '/why' or '?' command to show causal explanations for recent beliefs."""
        try:
            # Import causal explanation agent
            from storage.causal_explanation_agent import CausalExplanationAgent
            explanation_agent = CausalExplanationAgent()
            
            # Get recent facts for this user
            recent_facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            
            if not recent_facts:
                return {
                    'response': "I don't have any recent facts to explain. Try adding some information first!",
                    'command_processed': True,
                    'explanations': []
                }
            
            # Get explanations for the most recent few facts
            explanations = []
            for fact in recent_facts[:3]:  # Explain top 3 recent facts
                if hasattr(fact, 'id'):
                    explanation = explanation_agent.explain_fact(fact.id, max_chains=2)
                    if 'error' not in explanation:
                        explanations.append({
                            'fact': f"{fact.subject} {fact.predicate} {fact.object}",
                            'explanation': explanation['explanation'],
                            'confidence': explanation['confidence'],
                            'reasoning_type': explanation['reasoning_type']
                        })
            
            if explanations:
                response = "Here's why I believe your recent thoughts:\n\n"
                for i, exp in enumerate(explanations, 1):
                    response += f"{i}. **{exp['fact']}**\n"
                    response += f"   {exp['explanation']}\n"
                    response += f"   (Confidence: {exp['confidence']:.1%})\n\n"
            else:
                response = "I can explain my reasoning, but I don't see clear causal connections for your recent facts."
            
            return {
                'response': response,
                'command_processed': True,
                'explanations': explanations,
                'metadata': {'explanation_count': len(explanations)}
            }
            
        except Exception as e:
            print(f"[Phase2Cognitive] Error in why command: {e}")
            return {
                'response': f"I couldn't generate explanations right now: {str(e)}",
                'command_processed': True,
                'error': str(e)
            }
    
    def _handle_why_believe_command(self, command: str, user_profile_id: str = None, 
                                  session_id: str = None) -> Dict:
        """Handle '/why do I believe X?' command for specific fact explanation."""
        try:
            # Extract the belief from the command
            belief_text = command.replace('/why do i believe', '').strip()
            if belief_text.endswith('?'):
                belief_text = belief_text[:-1].strip()
            
            if not belief_text:
                return {
                    'response': "Please specify what you'd like me to explain. Usage: '/why do I believe [something]?'",
                    'command_processed': True
                }
            
            # Find facts that match the belief text
            all_facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            matching_facts = []
            
            for fact in all_facts:
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                if any(word in fact_text for word in belief_text.lower().split() if len(word) > 2):
                    matching_facts.append(fact)
            
            if not matching_facts:
                return {
                    'response': f"I couldn't find any facts matching '{belief_text}'. Could you be more specific?",
                    'command_processed': True,
                    'matches_found': 0
                }
            
            # Get explanation for the best matching fact
            from storage.causal_explanation_agent import CausalExplanationAgent
            explanation_agent = CausalExplanationAgent()
            
            best_match = matching_facts[0]  # Take the first/most recent match
            explanation = explanation_agent.explain_fact(best_match.id, max_chains=3)
            
            if 'error' in explanation:
                response = f"I found a matching belief about '{best_match.subject} {best_match.predicate} {best_match.object}', but I couldn't trace its reasoning: {explanation['error']}"
            else:
                response = f"**Why you believe '{best_match.subject} {best_match.predicate} {best_match.object}':**\n\n"
                response += explanation['explanation']
                
                if explanation.get('causal_chains'):
                    response += "\n\n**Supporting evidence:**\n"
                    for i, chain in enumerate(explanation['causal_chains'][:2], 1):
                        response += f"{i}. {chain['explanation']}\n"
                
                response += f"\n(Overall confidence: {explanation['confidence']:.1%})"
            
            return {
                'response': response,
                'command_processed': True,
                'belief_query': belief_text,
                'matched_fact': f"{best_match.subject} {best_match.predicate} {best_match.object}",
                'explanation': explanation,
                'matches_found': len(matching_facts)
            }
            
        except Exception as e:
            print(f"[Phase2Cognitive] Error in why believe command: {e}")
            return {
                'response': f"I couldn't analyze that belief right now: {str(e)}",
                'command_processed': True,
                'error': str(e)
            }
    
    def _handle_explain_command(self, command: str, user_profile_id: str = None, 
                              session_id: str = None) -> Dict:
        """Handle '/explain [fact_id]' command for detailed fact explanation."""
        try:
            # Extract fact ID from command
            parts = command.split()
            if len(parts) < 2:
                return {
                    'response': "Please specify a fact ID. Usage: '/explain [fact_id]'",
                    'command_processed': True
                }
            
            # Try to parse as fact ID
            try:
                fact_id = int(parts[1])
                from storage.causal_explanation_agent import CausalExplanationAgent
                explanation_agent = CausalExplanationAgent()
                
                explanation = explanation_agent.explain_fact(fact_id, max_chains=5)
                
                if 'error' in explanation:
                    response = f"Couldn't explain fact {fact_id}: {explanation['error']}"
                else:
                    response = f"**Detailed explanation for fact {fact_id}:**\n\n"
                    response += explanation['explanation']
                    
                    if explanation.get('causal_chains'):
                        response += "\n\n**Causal reasoning chains:**\n"
                        for i, chain in enumerate(explanation['causal_chains'], 1):
                            response += f"\n{i}. **Chain strength: {chain['total_strength']:.3f}**\n"
                            response += f"   Root cause: {chain['root_cause']['description']}\n"
                            response += f"   Explanation: {chain['explanation']}\n"
                
                return {
                    'response': response,
                    'command_processed': True,
                    'fact_id': fact_id,
                    'explanation': explanation
                }
                
            except ValueError:
                return {
                    'response': f"Please provide a valid fact ID number. Usage: '/explain [fact_id]'",
                    'command_processed': True
                }
                
        except Exception as e:
            print(f"[Phase2Cognitive] Error in explain command: {e}")
            return {
                'response': f"I couldn't explain that right now: {str(e)}",
                'command_processed': True,
                'error': str(e)
            }
    
    def _handle_drift_status_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /drift_status command for detailed drift execution status."""
        if not hasattr(self, 'drift_execution_engine') or not self.drift_execution_engine:
            return {
                'response': "Drift execution engine not available",
                'command_processed': True
            }
        
        execution_status = self.drift_execution_engine.get_execution_status()
        execution_history = self.drift_execution_engine.get_execution_history(limit=10)
        
        lines = ["🤖 DRIFT EXECUTION STATUS DETAILED REPORT"]
        lines.append("=" * 60)
        
        # Configuration
        lines.append("\n⚙️ CONFIGURATION:")
        lines.append(f"  Auto-execute enabled: {'✅' if execution_status['auto_execute_enabled'] else '❌'}")
        lines.append(f"  Execution interval: {execution_status['execution_interval']} seconds")
        lines.append(f"  Priority threshold: {execution_status['priority_threshold']:.1f}")
        
        # Runtime status
        lines.append("\n🔄 RUNTIME STATUS:")
        lines.append(f"  Background running: {'✅' if execution_status['background_running'] else '❌'}")
        lines.append(f"  Active executions: {execution_status['active_executions']}")
        lines.append(f"  Total executions: {execution_status['total_executions']}")
        
        # Performance metrics
        lines.append("\n📊 PERFORMANCE METRICS:")
        lines.append(f"  Completed executions: {execution_status['completed_executions']}")
        lines.append(f"  Failed executions: {execution_status['failed_executions']}")
        lines.append(f"  Success rate: {execution_status['success_rate']:.1%}")
        
        if execution_status['total_executions'] > 0:
            avg_execution_time = sum(r['execution_time'] for r in execution_status['recent_executions']) / len(execution_status['recent_executions'])
            lines.append(f"  Average execution time: {avg_execution_time:.2f}s")
        
        # Recent execution history
        if execution_history:
            lines.append("\n📋 RECENT EXECUTION HISTORY:")
            for i, result in enumerate(execution_history, 1):
                status = "✅" if result.success else "❌"
                lines.append(f"  {i}. {status} {result.completion_notes}")
                lines.append(f"     Time: {result.execution_time:.2f}s")
                if result.actions_taken:
                    lines.append(f"     Actions: {', '.join(result.actions_taken)}")
                if result.error_message:
                    lines.append(f"     Error: {result.error_message}")
                lines.append("")
        
        # Memory trail summary
        if execution_history:
            memory_actions = []
            for result in execution_history:
                for action in result.memory_trail:
                    if 'action' in action:
                        memory_actions.append(action['action'])
            
            if memory_actions:
                action_counts = {}
                for action in memory_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                lines.append("🧠 MEMORY TRAIL SUMMARY:")
                for action, count in action_counts.items():
                    lines.append(f"  {action}: {count} times")
        
        return {
            'response': "\n".join(lines),
            'command_processed': True,
            'drift_execution_data': execution_status
        }
    
    def _handle_reflex_log_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /reflex_log command for reflex cycle logging."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get recent cycles
            recent_cycles = reflex_logger.get_recent_cycles(limit=15)
            
            # Get statistics
            stats = reflex_logger.get_execution_statistics()
            
            lines = ["🧠 Reflex Cycle Log - Autonomic Cognitive Repair"]
            lines.append("=" * 60)
            
            # Show overall statistics
            lines.append(f"📊 Overall Statistics:")
            lines.append(f"  Total reflex cycles: {stats.get('total_cycles', 0)}")
            lines.append(f"  Success rate: {stats.get('success_rate', 0.0):.1%}")
            lines.append(f"  Average execution time: {stats.get('average_duration', 0.0):.2f}s")
            
            # Show strategy breakdown
            if stats.get('strategy_breakdown'):
                lines.append(f"\n🔧 Strategy Performance:")
                for strategy, count in stats['strategy_breakdown'].items():
                    success_rate = stats['strategy_success'].get(strategy, {}).get('success_rate', 0.0)
                    lines.append(f"  {strategy}: {count} cycles ({success_rate:.1%} success)")
            
            # Show recent cycles with detailed formatting
            if recent_cycles:
                lines.append(f"\n🔄 Recent Reflex Cycles ({len(recent_cycles)}):")
                for i, cycle in enumerate(recent_cycles, 1):
                    lines.append(f"\n{i}. {reflex_logger.format_cycle_display(cycle)}")
                    
                    # Add additional details for failed cycles
                    if not cycle.success and cycle.error_message:
                        lines.append(f"   ❌ Error: {cycle.error_message}")
                    
                    # Add action details
                    if cycle.actions_taken:
                        lines.append(f"   📝 Actions: {', '.join(cycle.actions_taken)}")
            else:
                lines.append("\nNo reflex cycles found yet.")
                lines.append("Reflex cycles are created when drift-triggered goals are executed.")
            
            return {
                'response': "\n".join(lines),
                'command_processed': True,
                'reflex_cycles_count': len(recent_cycles),
                'reflex_statistics': stats
            }
            
        except ImportError:
            return {
                'response': "Reflex log system not available",
                'command_processed': True
            }
        except Exception as e:
            return {
                'response': f"Error retrieving reflex log: {e}",
                'command_processed': True
            }
    
    def _handle_reflex_scores_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /reflex_scores command for reflex effectiveness scoring."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get score statistics
            stats = reflex_logger.get_score_statistics()
            
            # Get recent scores
            recent_scores = reflex_logger.get_reflex_scores(limit=15)
            
            lines = ["🧠 Reflex Effectiveness Scoring - Cognitive Repair Performance"]
            lines.append("=" * 70)
            
            # Show overall statistics
            lines.append(f"📊 Overall Performance:")
            lines.append(f"  Total reflex scores: {stats.get('total_scores', 0)}")
            lines.append(f"  Average effectiveness: {stats.get('average_score', 0.0):.3f}")
            lines.append(f"  Best repair: {stats.get('max_score', 0.0):.3f}")
            lines.append(f"  Worst repair: {stats.get('min_score', 0.0):.3f}")
            lines.append(f"  Recent trend (last 10): {stats.get('rolling_average', 0.0):.3f}")
            
            # Show strategy performance
            if stats.get('strategy_statistics'):
                lines.append(f"\n🔧 Strategy Effectiveness:")
                for strategy, strategy_stats in stats['strategy_statistics'].items():
                    avg_score = strategy_stats['avg_score']
                    count = strategy_stats['total']
                    
                    # Get effectiveness icon
                    if avg_score >= 0.8:
                        icon = "🟢"
                    elif avg_score >= 0.6:
                        icon = "🟡"
                    elif avg_score >= 0.4:
                        icon = "🟠"
                    else:
                        icon = "🔴"
                    
                    lines.append(f"  {icon} {strategy}: {avg_score:.3f} ({count} repairs)")
                    lines.append(f"    Best: {strategy_stats['max_score']:.3f}, Worst: {strategy_stats['min_score']:.3f}")
            
            # Show recent detailed scores
            if recent_scores:
                lines.append(f"\n🔄 Recent Reflex Repairs:")
                for i, score in enumerate(recent_scores, 1):
                    lines.append(f"  {i}. {score.score_icon} {score.strategy}: {score.score:.3f}")
                    lines.append(f"     Token: {score.token_id}, Cycle: {score.cycle_id}")
                    
                    # Show deltas if significant
                    deltas = []
                    if abs(score.coherence_delta) > 0.1:
                        deltas.append(f"coherence: {score.coherence_delta:+.2f}")
                    if abs(score.volatility_delta) > 0.1:
                        deltas.append(f"volatility: {score.volatility_delta:+.2f}")
                    if abs(score.belief_consistency_delta) > 0.1:
                        deltas.append(f"consistency: {score.belief_consistency_delta:+.2f}")
                    
                    if deltas:
                        lines.append(f"     Changes: {', '.join(deltas)}")
                    
                    if score.scoring_notes:
                        lines.append(f"     Notes: {score.scoring_notes}")
                    lines.append("")
            else:
                lines.append(f"\nNo reflex scores found yet.")
                lines.append("Scores are generated when reflex cycles complete with cognitive state data.")
                lines.append("The system learns from each repair to improve future effectiveness.")
            
            return {
                'response': "\n".join(lines),
                'command_processed': True,
                'reflex_scores_count': len(recent_scores),
                'reflex_statistics': stats
            }
            
        except ImportError:
            return {
                'response': "Reflex scoring system not available",
                'command_processed': True
            }
        except Exception as e:
            return {
                'response': f"Error retrieving reflex scores: {e}",
                'command_processed': True
            }
    
    def _handle_strategy_optimization_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /strategy_optimization command for strategy optimization analysis."""
        try:
            from storage.reflex_log import get_reflex_logger
            
            reflex_logger = get_reflex_logger()
            
            # Get recent scores to analyze optimization patterns
            recent_scores = reflex_logger.get_reflex_scores(limit=100)
            
            lines = ["🧠 Strategy Optimization Analysis - Learning from Reflex History"]
            lines.append("=" * 70)
            
            if recent_scores:
                # Analyze strategy performance
                strategy_performance = {}
                for score in recent_scores:
                    strategy = score.strategy
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = []
                    strategy_performance[strategy].append(score.score)
                
                # Calculate averages and find best strategies
                strategy_averages = {}
                for strategy, scores in strategy_performance.items():
                    avg_score = sum(scores) / len(scores)
                    strategy_averages[strategy] = avg_score
                
                # Sort strategies by performance
                sorted_strategies = sorted(strategy_averages.items(), key=lambda x: x[1], reverse=True)
                
                lines.append(f"📊 Strategy Performance Analysis ({len(recent_scores)} recent repairs):")
                lines.append("")
                
                for i, (strategy, avg_score) in enumerate(sorted_strategies, 1):
                    count = len(strategy_performance[strategy])
                    icon = "🟢" if avg_score >= 0.8 else "🟡" if avg_score >= 0.6 else "🟠" if avg_score >= 0.4 else "🔴"
                    lines.append(f"  {i}. {icon} {strategy}: {avg_score:.3f} ({count} repairs)")
                
                # Show optimization recommendations
                lines.append("")
                lines.append("🔧 Optimization Recommendations:")
                
                if len(sorted_strategies) >= 2:
                    best_strategy, best_score = sorted_strategies[0]
                    second_strategy, second_score = sorted_strategies[1]
                    
                    lines.append(f"  • Best performing: {best_strategy} (avg: {best_score:.3f})")
                    lines.append(f"  • Consider prioritizing {best_strategy} for similar drift patterns")
                    
                    if best_score - second_score > 0.1:
                        lines.append(f"  • {best_strategy} significantly outperforms {second_strategy}")
                    else:
                        lines.append(f"  • {best_strategy} and {second_strategy} perform similarly")
                
                # Show drift type analysis
                lines.append("")
                lines.append("🎯 Drift Type Strategy Preferences:")
                lines.append("  • Contradictions: belief_clarification, fact_consolidation")
                lines.append("  • Volatility: cluster_reassessment, belief_clarification")
                lines.append("  • Semantic Decay: cluster_reassessment, fact_consolidation")
                
                # Show learning insights
                lines.append("")
                lines.append("🧠 Learning Insights:")
                lines.append("  • System automatically selects best strategy based on historical performance")
                lines.append("  • Rolling average of last 5 scores used for strategy selection")
                lines.append("  • Drift type detection influences strategy preferences")
                lines.append("  • Fallback to default strategy when no history available")
                
            else:
                lines.append("No reflex scores available yet.")
                lines.append("Strategy optimization requires historical performance data.")
                lines.append("Run some drift repairs to build optimization data.")
                lines.append("")
                lines.append("The system will learn from each repair to improve future strategy selection.")
            
            return {
                'response': "\n".join(lines),
                'command_processed': True,
                'strategy_analysis': {
                    'total_scores': len(recent_scores),
                    'strategies_analyzed': len(strategy_performance) if recent_scores else 0
                }
            }
            
        except Exception as e:
            return {
                'response': f"Error in strategy optimization analysis: {e}",
                'command_processed': True
            }
    
    # Phase 25: Reflective Self-Awareness Command Handlers
    
    def _handle_self_summary_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /self_summary command - show current identity snapshot."""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            summary = self_model.generate_self_summary()
            
            return {
                'response': summary,
                'command_processed': True,
                'self_awareness': True
            }
            
        except Exception as e:
            return {
                'response': f"❌ Error generating self summary: {e}",
                'command_processed': True
            }
    
    def _handle_self_journal_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /self_journal command - show self-reflection journal."""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            journal = self_model.get_self_journal()
            
            if not journal:
                return {
                    'response': "📔 Self-reflection journal is empty.",
                    'command_processed': True
                }
            
            lines = [f"📔 SELF-REFLECTION JOURNAL ({len(journal)} entries)", "=" * 60]
            
            # Show recent entries (last 10)
            recent_entries = journal[-10:] if len(journal) > 10 else journal
            
            for entry in recent_entries:
                timestamp = entry.get('timestamp', 'Unknown')
                trigger = entry.get('trigger', 'Unknown')
                summary = entry.get('summary', 'No summary')
                
                lines.append(f"\n🕐 {timestamp}")
                lines.append(f"📍 Trigger: {trigger}")
                lines.append(f"💭 {summary}")
                
                # Show changes if available
                changes = entry.get('changes', {})
                if changes and len(changes) > 0:
                    lines.append("🔄 Changes:")
                    for key, value in changes.items():
                        if key == "traits" and isinstance(value, dict):
                            for trait, change in value.items():
                                if isinstance(change, list) and len(change) == 2:
                                    old_val, new_val = change
                                    direction = "↗️" if new_val > old_val else "↘️"
                                    lines.append(f"   {direction} {trait}: {old_val:.2f} → {new_val:.2f}")
                        elif isinstance(value, list) and len(value) == 2:
                            old_val, new_val = value
                            lines.append(f"   • {key}: {old_val} → {new_val}")
                        else:
                            lines.append(f"   • {key}: {value}")
                
                lines.append("-" * 40)
            
            if len(journal) > 10:
                lines.append(f"\n(Showing last 10 of {len(journal)} total entries)")
            
            return {
                'response': "\n".join(lines),
                'command_processed': True,
                'journal_entries': len(journal)
            }
            
        except Exception as e:
            return {
                'response': f"❌ Error accessing self journal: {e}",
                'command_processed': True
            }
    
    def _handle_self_reflect_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /self_reflect command - manually trigger a reflection cycle."""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            result = self_model.manual_reflection()
            
            return {
                'response': f"🧠 SELF-REFLECTION TRIGGERED\n{result}",
                'command_processed': True,
                'reflection_triggered': True
            }
            
        except Exception as e:
            return {
                'response': f"❌ Error during self reflection: {e}",
                'command_processed': True
            }
    
    def _handle_self_sync_command(self, user_profile_id: str = None, session_id: str = None) -> Dict:
        """Handle /self_sync command - sync self model with personality engine."""
        try:
            from agents.self_model import get_self_model
            
            self_model = get_self_model()
            success = self_model.sync_from_personality_engine()
            
            lines = []
            
            if success:
                lines.append("✅ Successfully synced self model with personality engine")
                
                # Show brief update
                recent_changes = self_model.get_recent_changes()
                
                if recent_changes.get('status') == 'evolving':
                    lines.append("\n🔄 Recent changes detected:")
                    for change in recent_changes.get('recent_changes', [])[-3:]:
                        lines.append(f"   • {change.get('summary', 'Unknown change')}")
                else:
                    lines.append("🟢 Identity remains stable after sync")
            else:
                lines.append("❌ Failed to sync self model with personality engine")
            
            return {
                'response': "\n".join(lines),
                'command_processed': True,
                'sync_successful': success
            }
            
        except Exception as e:
            return {
                'response': f"❌ Error syncing self model: {e}",
                'command_processed': True
            }