#!/usr/bin/env python3
"""
PlanMemory - Storage and retrieval system for recursive plans

Manages plan persistence, similarity matching, outcome tracking,
and plan evolution within MeRNSTA's memory architecture.
"""

import sqlite3
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib

# Import Plan class from recursive_planner
try:
    from agents.recursive_planner import Plan, PlanStep
    PLAN_CLASSES_AVAILABLE = True
except ImportError:
    PLAN_CLASSES_AVAILABLE = False
    logging.warning("Plan classes not available - using dict representations")

from storage.db_utils import get_conn
from config.settings import get_config

class PlanMemory:
    """
    Storage system for managing recursive plans with similarity matching.
    
    Capabilities:
    - Plan persistence with full metadata
    - Similarity-based plan retrieval
    - Outcome tracking and success metrics
    - Plan evolution and versioning
    - Integration with existing memory systems
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config().get('recursive_planning', {})
        self.db_path = db_path or self.config.get('plan_db_path', 'plan_memory.db')
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.max_similar_plans = self.config.get('max_similar_plans', 10)
        
        # Initialize database
        self._init_database()
        
        # Initialize embedding system for similarity
        self._embedding_system = None
        
        logging.info(f"[PlanMemory] Initialized with db: {self.db_path}")
    
    @property
    def embedding_system(self):
        """Lazy-load embedding system for similarity calculations"""
        if self._embedding_system is None:
            try:
                from scripts.embedder import get_embeddings_cached
                self._embedding_system = get_embeddings_cached
            except ImportError:
                logging.warning("[PlanMemory] Embedding system not available - using text similarity")
                self._embedding_system = self._text_similarity_fallback
        return self._embedding_system
    
    def _init_database(self):
        """Initialize plan storage database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Plans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                plan_id TEXT PRIMARY KEY,
                goal_text TEXT NOT NULL,
                plan_type TEXT DEFAULT 'sequential',
                status TEXT DEFAULT 'draft',
                confidence REAL DEFAULT 0.8,
                priority INTEGER DEFAULT 1,
                parent_goal_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                success_criteria TEXT,  -- JSON array
                risk_factors TEXT,      -- JSON array
                intention_chain TEXT,   -- JSON array
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_execution_time REAL DEFAULT 0.0,
                goal_embedding TEXT,    -- Embedding vector for similarity
                goal_hash TEXT,         -- Hash for quick lookup
                metadata TEXT,          -- Additional JSON metadata
                FOREIGN KEY (parent_goal_id) REFERENCES plans (plan_id)
            )
        """)
        
        # Plan steps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_steps (
                step_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                step_order INTEGER NOT NULL,
                subgoal TEXT NOT NULL,
                why TEXT NOT NULL,
                expected_result TEXT NOT NULL,
                prerequisites TEXT,     -- JSON array
                resources_needed TEXT,  -- JSON array
                status TEXT DEFAULT 'pending',
                confidence REAL DEFAULT 0.8,
                priority INTEGER DEFAULT 1,
                estimated_duration TEXT,
                actual_duration REAL,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                subplan_id TEXT,        -- Reference to sub-plan if exists
                metadata TEXT,          -- Additional JSON metadata
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (subplan_id) REFERENCES plans (plan_id)
            )
        """)
        
        # Plan executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_executions (
                execution_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,   -- running, completed, failed, aborted
                completion_percentage REAL DEFAULT 0.0,
                overall_success BOOLEAN DEFAULT FALSE,
                steps_executed INTEGER DEFAULT 0,
                steps_failed INTEGER DEFAULT 0,
                execution_log TEXT,     -- JSON array of log entries
                results TEXT,           -- JSON execution results
                user_profile_id TEXT,
                session_id TEXT,
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id)
            )
        """)
        
        # Plan similarities table (for caching similarity calculations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_similarities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id_1 TEXT NOT NULL,
                plan_id_2 TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                similarity_type TEXT DEFAULT 'embedding',  -- embedding, text, semantic
                calculated_at TEXT NOT NULL,
                FOREIGN KEY (plan_id_1) REFERENCES plans (plan_id),
                FOREIGN KEY (plan_id_2) REFERENCES plans (plan_id),
                UNIQUE(plan_id_1, plan_id_2, similarity_type)
            )
        """)
        
        # Phase 16: DAG and dependency support tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                dependency_step_id TEXT NOT NULL,
                dependency_type TEXT DEFAULT 'prerequisite',  -- prerequisite, conditional, fallback
                condition_logic TEXT,                         -- JSON condition for conditional dependencies
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (step_id) REFERENCES plan_steps (step_id),
                FOREIGN KEY (dependency_step_id) REFERENCES plan_steps (step_id),
                UNIQUE(step_id, dependency_step_id, dependency_type)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_plan_id TEXT NOT NULL,
                target_plan_id TEXT NOT NULL,
                chain_type TEXT DEFAULT 'sequential',     -- sequential, conditional, parallel
                trigger_condition TEXT,                   -- JSON condition for activation
                priority INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (target_plan_id) REFERENCES plans (plan_id),
                UNIQUE(source_plan_id, target_plan_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                branch_condition TEXT NOT NULL,           -- JSON condition
                alternative_steps TEXT NOT NULL,          -- JSON array of alternative step IDs
                branch_priority INTEGER DEFAULT 1,
                activation_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (step_id) REFERENCES plan_steps (step_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                source_step_id TEXT NOT NULL,
                target_step_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,          -- next_step, fallback_step, alternative
                weight REAL DEFAULT 1.0,                  -- Relationship strength/priority
                metadata TEXT,                            -- JSON metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (source_step_id) REFERENCES plan_steps (step_id),
                FOREIGN KEY (target_step_id) REFERENCES plan_steps (step_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                checkpoint_name TEXT NOT NULL,
                checkpoint_type TEXT DEFAULT 'progress',  -- progress, validation, milestone
                validation_criteria TEXT,                 -- JSON validation criteria
                reached_at TEXT,                          -- Timestamp when reached
                metadata TEXT,                            -- JSON metadata
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id),
                FOREIGN KEY (step_id) REFERENCES plan_steps (step_id)
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_goal_hash ON plans (goal_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_status ON plans (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plans_parent ON plans (parent_goal_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_plan ON plan_steps (plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_plan ON plan_executions (plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_similarities_plan1 ON plan_similarities (plan_id_1)")
        
        # Phase 16 indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dependencies_step ON plan_dependencies (step_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dependencies_dep_step ON plan_dependencies (dependency_step_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chains_source ON plan_chains (source_plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chains_target ON plan_chains (target_plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_branches_step ON plan_branches (step_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON step_relationships (source_step_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON step_relationships (target_step_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_plan ON plan_checkpoints (plan_id)")
        
        conn.commit()
        conn.close()
        
        logging.info("[PlanMemory] Database schema initialized")
    
    def store_plan(self, plan) -> bool:
        """
        Store a plan in the database.
        
        Args:
            plan: Plan object or dict to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert plan to dict if needed
            if PLAN_CLASSES_AVAILABLE and isinstance(plan, Plan):
                plan_dict = self._plan_to_dict(plan)
            else:
                plan_dict = plan
            
            # Generate goal hash for similarity
            goal_hash = self._generate_goal_hash(plan_dict['goal_text'])
            
            # Get embedding for similarity matching
            goal_embedding = self._get_goal_embedding(plan_dict['goal_text'])
            
            # Insert plan record
            cursor.execute("""
                INSERT OR REPLACE INTO plans (
                    plan_id, goal_text, plan_type, status, confidence, priority,
                    parent_goal_id, created_at, updated_at, success_criteria,
                    risk_factors, intention_chain, goal_embedding, goal_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan_dict['plan_id'],
                plan_dict['goal_text'],
                plan_dict.get('plan_type', 'sequential'),
                plan_dict.get('status', 'draft'),
                plan_dict.get('confidence', 0.8),
                plan_dict.get('priority', 1),
                plan_dict.get('parent_goal_id'),
                plan_dict.get('created_at', datetime.now().isoformat()),
                plan_dict.get('updated_at', datetime.now().isoformat()),
                json.dumps(plan_dict.get('success_criteria', [])),
                json.dumps(plan_dict.get('risk_factors', [])),
                json.dumps(plan_dict.get('intention_chain', [])),
                goal_embedding,
                goal_hash,
                json.dumps(plan_dict.get('metadata', {}))
            ))
            
            # Insert plan steps
            for i, step in enumerate(plan_dict.get('steps', [])):
                if PLAN_CLASSES_AVAILABLE and hasattr(step, '__dict__'):
                    step_dict = step.__dict__
                else:
                    step_dict = step
                
                cursor.execute("""
                    INSERT OR REPLACE INTO plan_steps (
                        step_id, plan_id, step_order, subgoal, why, expected_result,
                        prerequisites, resources_needed, status, confidence, priority,
                        estimated_duration, subplan_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    step_dict.get('step_id', str(uuid.uuid4())),
                    plan_dict['plan_id'],
                    i,
                    step_dict.get('subgoal', ''),
                    step_dict.get('why', ''),
                    step_dict.get('expected_result', ''),
                    json.dumps(step_dict.get('prerequisites', [])),
                    json.dumps(step_dict.get('resources_needed', [])),
                    step_dict.get('status', 'pending'),
                    step_dict.get('confidence', 0.8),
                    step_dict.get('priority', 1),
                    step_dict.get('estimated_duration'),
                    step_dict.get('subplan_id'),
                    json.dumps(step_dict.get('metadata', {}))
                ))
            
            # Phase 16: Store DAG relationships and dependencies
            self._store_plan_relationships(cursor, plan_dict)
            
            # Store plan chains if present
            for chained_plan_id in plan_dict.get('chained_plans', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO plan_chains (
                        source_plan_id, target_plan_id, chain_type, priority
                    ) VALUES (?, ?, ?, ?)
                """, (
                    plan_dict['plan_id'],
                    chained_plan_id,
                    'sequential',
                    1
                ))
            
            # Store branching points
            for step_id, alternatives in plan_dict.get('branching_points', {}).items():
                cursor.execute("""
                    INSERT OR REPLACE INTO plan_branches (
                        plan_id, step_id, branch_condition, alternative_steps, branch_priority
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    plan_dict['plan_id'],
                    step_id,
                    json.dumps({"alternatives": alternatives}),
                    json.dumps(alternatives),
                    1
                ))
            
            # Store progress checkpoints
            for checkpoint_step_id in plan_dict.get('progress_checkpoints', []):
                cursor.execute("""
                    INSERT OR REPLACE INTO plan_checkpoints (
                        plan_id, step_id, checkpoint_name, checkpoint_type
                    ) VALUES (?, ?, ?, ?)
                """, (
                    plan_dict['plan_id'],
                    checkpoint_step_id,
                    f"Checkpoint_{checkpoint_step_id}",
                    'progress'
                ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Stored plan with DAG support: {plan_dict['plan_id']}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error storing plan: {e}")
            return False
    
    def get_similar_plans(self, goal_text: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find plans similar to the given goal text.
        
        Args:
            goal_text: Goal to find similar plans for
            limit: Maximum number of plans to return
            
        Returns:
            List of similar plans with similarity scores
        """
        try:
            limit = limit or self.max_similar_plans
            
            # Get embedding for the goal
            goal_embedding = self._get_goal_embedding(goal_text)
            goal_hash = self._generate_goal_hash(goal_text)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First try exact hash match
            cursor.execute("""
                SELECT plan_id, goal_text, status, execution_count, success_count, failure_count
                FROM plans 
                WHERE goal_hash = ? AND status != 'draft'
                ORDER BY success_count DESC, execution_count DESC
                LIMIT ?
            """, (goal_hash, limit))
            
            exact_matches = cursor.fetchall()
            
            if exact_matches:
                results = []
                for row in exact_matches:
                    results.append({
                        'plan_id': row[0],
                        'goal_text': row[1],
                        'status': row[2],
                        'execution_count': row[3],
                        'success_count': row[4],
                        'failure_count': row[5],
                        'similarity_score': 1.0,  # Exact match
                        'success_rate': row[4] / max(1, row[3])
                    })
                
                conn.close()
                logging.info(f"[PlanMemory] Found {len(results)} exact matches for goal")
                return results
            
            # If no exact matches, use similarity
            cursor.execute("""
                SELECT plan_id, goal_text, goal_embedding, status, 
                       execution_count, success_count, failure_count
                FROM plans 
                WHERE status != 'draft' AND goal_embedding IS NOT NULL
                ORDER BY execution_count DESC
                LIMIT 50
            """)
            
            candidates = cursor.fetchall()
            conn.close()
            
            if not candidates:
                logging.info("[PlanMemory] No candidate plans found for similarity matching")
                return []
            
            # Calculate similarities
            similar_plans = []
            for row in candidates:
                plan_id, plan_goal, plan_embedding, status, exec_count, success_count, failure_count = row
                
                similarity = self._calculate_similarity(goal_embedding, plan_embedding, goal_text, plan_goal)
                
                if similarity >= self.similarity_threshold:
                    similar_plans.append({
                        'plan_id': plan_id,
                        'goal_text': plan_goal,
                        'status': status,
                        'execution_count': exec_count,
                        'success_count': success_count,
                        'failure_count': failure_count,
                        'similarity_score': similarity,
                        'success_rate': success_count / max(1, exec_count)
                    })
            
            # Sort by combined score (similarity + success rate)
            similar_plans.sort(key=lambda x: (x['similarity_score'] * 0.6 + x['success_rate'] * 0.4), reverse=True)
            
            result = similar_plans[:limit]
            logging.info(f"[PlanMemory] Found {len(result)} similar plans for goal")
            
            return result
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error finding similar plans: {e}")
            return []
    
    def record_plan_outcome(self, plan_id: str, result: Dict[str, Any]) -> bool:
        """
        Record the outcome of a plan execution.
        
        Args:
            plan_id: ID of the executed plan
            result: Execution result dictionary
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert execution record
            execution_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO plan_executions (
                    execution_id, plan_id, started_at, completed_at, status,
                    completion_percentage, overall_success, steps_executed,
                    steps_failed, execution_log, results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                plan_id,
                result.get('execution_start', datetime.now().isoformat()),
                result.get('execution_end', datetime.now().isoformat()),
                'completed' if result.get('overall_success', False) else 'failed',
                result.get('completion_percentage', 0.0),
                result.get('overall_success', False),
                len(result.get('steps_executed', [])),
                len(result.get('steps_failed', [])),
                json.dumps(result.get('execution_log', [])),
                json.dumps(result)
            ))
            
            # Update plan statistics
            success_increment = 1 if result.get('overall_success', False) else 0
            failure_increment = 0 if result.get('overall_success', False) else 1
            
            cursor.execute("""
                UPDATE plans SET
                    execution_count = execution_count + 1,
                    success_count = success_count + ?,
                    failure_count = failure_count + ?,
                    updated_at = ?
                WHERE plan_id = ?
            """, (success_increment, failure_increment, datetime.now().isoformat(), plan_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Recorded outcome for plan: {plan_id}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error recording plan outcome: {e}")
            return False
    
    def get_plan_by_id(self, plan_id: str):
        """
        Retrieve a complete plan by ID.
        
        Args:
            plan_id: ID of the plan to retrieve
            
        Returns:
            Plan object or dict, None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get plan details
            cursor.execute("""
                SELECT plan_id, goal_text, plan_type, status, confidence, priority,
                       parent_goal_id, created_at, updated_at, success_criteria,
                       risk_factors, intention_chain, metadata
                FROM plans WHERE plan_id = ?
            """, (plan_id,))
            
            plan_row = cursor.fetchone()
            if not plan_row:
                conn.close()
                return None
            
            # Get plan steps
            cursor.execute("""
                SELECT step_id, subgoal, why, expected_result, prerequisites,
                       resources_needed, status, confidence, priority,
                       estimated_duration, subplan_id
                FROM plan_steps 
                WHERE plan_id = ? 
                ORDER BY step_order
            """, (plan_id,))
            
            step_rows = cursor.fetchall()
            conn.close()
            
            # Build plan dict
            plan_dict = {
                'plan_id': plan_row[0],
                'goal_text': plan_row[1],
                'plan_type': plan_row[2],
                'status': plan_row[3],
                'confidence': plan_row[4],
                'priority': plan_row[5],
                'parent_goal_id': plan_row[6],
                'created_at': plan_row[7],
                'updated_at': plan_row[8],
                'success_criteria': json.loads(plan_row[9] or '[]'),
                'risk_factors': json.loads(plan_row[10] or '[]'),
                'intention_chain': json.loads(plan_row[11] or '[]'),
                'metadata': json.loads(plan_row[12] or '{}'),
                'steps': []
            }
            
            # Build steps
            for step_row in step_rows:
                step_dict = {
                    'step_id': step_row[0],
                    'subgoal': step_row[1],
                    'why': step_row[2],
                    'expected_result': step_row[3],
                    'prerequisites': json.loads(step_row[4] or '[]'),
                    'resources_needed': json.loads(step_row[5] or '[]'),
                    'status': step_row[6],
                    'confidence': step_row[7],
                    'priority': step_row[8],
                    'estimated_duration': step_row[9],
                    'subplan_id': step_row[10]
                }
                plan_dict['steps'].append(step_dict)
            
            # Convert to Plan object if available
            if PLAN_CLASSES_AVAILABLE:
                return self._dict_to_plan(plan_dict)
            else:
                return plan_dict
                
        except Exception as e:
            logging.error(f"[PlanMemory] Error retrieving plan {plan_id}: {e}")
            return None
    
    def get_plans_by_status(self, status: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get plans filtered by status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT plan_id, goal_text, plan_type, confidence, priority,
                       created_at, updated_at, execution_count, success_count
                FROM plans 
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (status, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'plan_id': row[0],
                    'goal_text': row[1],
                    'plan_type': row[2],
                    'confidence': row[3],
                    'priority': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'execution_count': row[7],
                    'success_count': row[8]
                }
                for row in rows
            ]
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting plans by status: {e}")
            return []
    
    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan and all associated data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete in order due to foreign key constraints
            cursor.execute("DELETE FROM plan_executions WHERE plan_id = ?", (plan_id,))
            cursor.execute("DELETE FROM plan_similarities WHERE plan_id_1 = ? OR plan_id_2 = ?", (plan_id, plan_id))
            cursor.execute("DELETE FROM plan_steps WHERE plan_id = ?", (plan_id,))
            cursor.execute("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Deleted plan: {plan_id}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error deleting plan: {e}")
            return False
    
    def get_plan_statistics(self) -> Dict[str, Any]:
        """Get overall plan storage statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM plans")
            total_plans = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM plan_executions")
            total_executions = cursor.fetchone()[0]
            
            # Status breakdown
            cursor.execute("SELECT status, COUNT(*) FROM plans GROUP BY status")
            status_counts = dict(cursor.fetchall())
            
            # Success rates
            cursor.execute("""
                SELECT 
                    AVG(CASE WHEN execution_count > 0 THEN success_count * 1.0 / execution_count ELSE 0 END) as avg_success_rate,
                    COUNT(CASE WHEN execution_count > 0 THEN 1 END) as executed_plans
                FROM plans
            """)
            success_data = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_plans': total_plans,
                'total_executions': total_executions,
                'status_breakdown': status_counts,
                'average_success_rate': success_data[0] or 0.0,
                'executed_plans': success_data[1] or 0
            }
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting statistics: {e}")
            return {}
    
    def _plan_to_dict(self, plan) -> Dict[str, Any]:
        """Convert Plan object to dictionary"""
        plan_dict = {
            'plan_id': plan.plan_id,
            'goal_text': plan.goal_text,
            'plan_type': plan.plan_type,
            'status': plan.status,
            'confidence': plan.confidence,
            'priority': plan.priority,
            'parent_goal_id': plan.parent_goal_id,
            'created_at': plan.created_at,
            'updated_at': plan.updated_at,
            'success_criteria': plan.success_criteria,
            'risk_factors': plan.risk_factors,
            'intention_chain': plan.intention_chain,
            'steps': []
        }
        
        for step in plan.steps:
            step_dict = {
                'step_id': step.step_id,
                'subgoal': step.subgoal,
                'why': step.why,
                'expected_result': step.expected_result,
                'prerequisites': step.prerequisites,
                'resources_needed': step.resources_needed,
                'status': step.status,
                'confidence': step.confidence,
                'priority': step.priority,
                'estimated_duration': step.estimated_duration
            }
            plan_dict['steps'].append(step_dict)
        
        return plan_dict
    
    def _dict_to_plan(self, plan_dict: Dict[str, Any]):
        """Convert dictionary to Plan object"""
        from agents.recursive_planner import Plan, PlanStep
        
        steps = []
        for step_dict in plan_dict.get('steps', []):
            step = PlanStep(
                step_id=step_dict.get('step_id', str(uuid.uuid4())),
                subgoal=step_dict.get('subgoal', ''),
                why=step_dict.get('why', ''),
                expected_result=step_dict.get('expected_result', ''),
                prerequisites=step_dict.get('prerequisites', []),
                status=step_dict.get('status', 'pending'),
                confidence=step_dict.get('confidence', 0.8),
                priority=step_dict.get('priority', 1),
                estimated_duration=step_dict.get('estimated_duration'),
                resources_needed=step_dict.get('resources_needed', [])
            )
            steps.append(step)
        
        plan = Plan(
            plan_id=plan_dict['plan_id'],
            goal_text=plan_dict['goal_text'],
            steps=steps,
            plan_type=plan_dict.get('plan_type', 'sequential'),
            created_at=plan_dict.get('created_at'),
            updated_at=plan_dict.get('updated_at'),
            status=plan_dict.get('status', 'draft'),
            confidence=plan_dict.get('confidence', 0.8),
            priority=plan_dict.get('priority', 1),
            parent_goal_id=plan_dict.get('parent_goal_id'),
            intention_chain=plan_dict.get('intention_chain', []),
            success_criteria=plan_dict.get('success_criteria', []),
            risk_factors=plan_dict.get('risk_factors', [])
        )
        
        return plan
    
    def _generate_goal_hash(self, goal_text: str) -> str:
        """Generate a hash for goal text to enable quick exact matching"""
        # Normalize the goal text for consistent hashing
        normalized = goal_text.lower().strip()
        # Remove common articles and prepositions for better matching
        words = normalized.split()
        filtered_words = [w for w in words if w not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}]
        normalized = ' '.join(filtered_words)
        
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _get_goal_embedding(self, goal_text: str) -> Optional[str]:
        """Get embedding vector for goal text"""
        try:
            if callable(self.embedding_system):
                embedding = self.embedding_system([goal_text])
                if embedding and len(embedding) > 0:
                    return json.dumps(embedding[0])
            return None
        except Exception as e:
            logging.warning(f"[PlanMemory] Could not get embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: Optional[str], embedding2: Optional[str], 
                            text1: str, text2: str) -> float:
        """Calculate similarity between two goals"""
        try:
            # Try embedding-based similarity first
            if embedding1 and embedding2:
                vec1 = json.loads(embedding1)
                vec2 = json.loads(embedding2)
                return self._cosine_similarity(vec1, vec2)
            
            # Fallback to text similarity
            return self._text_similarity_fallback(text1, text2)
            
        except Exception as e:
            logging.warning(f"[PlanMemory] Similarity calculation error: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _text_similarity_fallback(self, text1: str, text2: str) -> float:
        """Fallback text similarity when embeddings unavailable"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    # === Phase 16: DAG and Dependency Support Methods ===
    
    def _store_plan_relationships(self, cursor, plan_dict: Dict[str, Any]) -> None:
        """Store step relationships for Phase 16 enhanced plans."""
        plan_id = plan_dict['plan_id']
        
        # Store step relationships (next_step, fallback_step, etc.)
        for step in plan_dict.get('steps', []):
            if PLAN_CLASSES_AVAILABLE and hasattr(step, '__dict__'):
                step_dict = step.__dict__
            else:
                step_dict = step
            
            step_id = step_dict.get('step_id')
            
            # Store next_step relationships
            if step_dict.get('next_step'):
                cursor.execute("""
                    INSERT OR REPLACE INTO step_relationships (
                        plan_id, source_step_id, target_step_id, relationship_type, weight
                    ) VALUES (?, ?, ?, ?, ?)
                """, (plan_id, step_id, step_dict['next_step'], 'next_step', 1.0))
            
            # Store fallback_step relationships
            if step_dict.get('fallback_step'):
                cursor.execute("""
                    INSERT OR REPLACE INTO step_relationships (
                        plan_id, source_step_id, target_step_id, relationship_type, weight
                    ) VALUES (?, ?, ?, ?, ?)
                """, (plan_id, step_id, step_dict['fallback_step'], 'fallback_step', 0.8))
            
            # Store conditional logic if present
            if step_dict.get('conditional_logic'):
                cursor.execute("""
                    INSERT OR REPLACE INTO step_relationships (
                        plan_id, source_step_id, target_step_id, relationship_type, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                """, (plan_id, step_id, step_id, 'conditional_logic', step_dict['conditional_logic']))
    
    def get_plan_dag(self, plan_id: str) -> Dict[str, Any]:
        """
        Get plan as a DAG structure with dependencies and relationships.
        
        Args:
            plan_id: ID of the plan to retrieve as DAG
            
        Returns:
            DAG representation with nodes, edges, and metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get plan info
            cursor.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,))
            plan_row = cursor.fetchone()
            
            if not plan_row:
                return {"error": "Plan not found"}
            
            plan_dict = dict(plan_row)
            
            # Get steps
            cursor.execute("""
                SELECT * FROM plan_steps 
                WHERE plan_id = ? 
                ORDER BY step_order
            """, (plan_id,))
            
            steps = [dict(row) for row in cursor.fetchall()]
            
            # Get relationships
            cursor.execute("""
                SELECT * FROM step_relationships 
                WHERE plan_id = ?
            """, (plan_id,))
            
            relationships = [dict(row) for row in cursor.fetchall()]
            
            # Get dependencies
            cursor.execute("""
                SELECT * FROM plan_dependencies 
                WHERE plan_id = ?
            """, (plan_id,))
            
            dependencies = [dict(row) for row in cursor.fetchall()]
            
            # Get branches
            cursor.execute("""
                SELECT * FROM plan_branches 
                WHERE plan_id = ?
            """, (plan_id,))
            
            branches = [dict(row) for row in cursor.fetchall()]
            
            # Get checkpoints
            cursor.execute("""
                SELECT * FROM plan_checkpoints 
                WHERE plan_id = ?
            """, (plan_id,))
            
            checkpoints = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            # Build DAG structure
            dag = {
                "plan_id": plan_id,
                "goal_text": plan_dict["goal_text"],
                "plan_type": plan_dict["plan_type"],
                "status": plan_dict["status"],
                "nodes": [],
                "edges": [],
                "dependencies": dependencies,
                "branches": branches,
                "checkpoints": checkpoints,
                "metadata": {
                    "created_at": plan_dict["created_at"],
                    "updated_at": plan_dict["updated_at"],
                    "confidence": plan_dict["confidence"]
                }
            }
            
            # Add nodes (steps)
            for step in steps:
                node = {
                    "id": step["step_id"],
                    "label": step["subgoal"],
                    "type": "step",
                    "status": step["status"],
                    "confidence": step["confidence"],
                    "priority": step["priority"],
                    "estimated_duration": step["estimated_duration"],
                    "prerequisites": json.loads(step["prerequisites"] or "[]"),
                    "resources_needed": json.loads(step["resources_needed"] or "[]")
                }
                dag["nodes"].append(node)
            
            # Add edges (relationships)
            for rel in relationships:
                edge = {
                    "from": rel["source_step_id"],
                    "to": rel["target_step_id"],
                    "type": rel["relationship_type"],
                    "weight": rel["weight"]
                }
                dag["edges"].append(edge)
            
            # Add dependency edges
            for dep in dependencies:
                edge = {
                    "from": dep["dependency_step_id"],
                    "to": dep["step_id"],
                    "type": f"dependency_{dep['dependency_type']}",
                    "weight": 1.0
                }
                dag["edges"].append(edge)
            
            return dag
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting plan DAG: {e}")
            return {"error": str(e)}
    
    def add_plan_dependency(self, plan_id: str, step_id: str, dependency_step_id: str, 
                           dependency_type: str = "prerequisite", condition_logic: Optional[str] = None) -> bool:
        """
        Add a dependency between two steps in a plan.
        
        Args:
            plan_id: Plan containing the steps
            step_id: Step that depends on another
            dependency_step_id: Step that is depended upon
            dependency_type: Type of dependency (prerequisite, conditional, fallback)
            condition_logic: Optional JSON condition for conditional dependencies
            
        Returns:
            True if dependency added successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO plan_dependencies (
                    plan_id, step_id, dependency_step_id, dependency_type, condition_logic
                ) VALUES (?, ?, ?, ?, ?)
            """, (plan_id, step_id, dependency_step_id, dependency_type, condition_logic))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Added dependency: {step_id} -> {dependency_step_id} ({dependency_type})")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error adding dependency: {e}")
            return False
    
    def remove_plan_dependency(self, plan_id: str, step_id: str, dependency_step_id: str) -> bool:
        """
        Remove a dependency between two steps.
        
        Args:
            plan_id: Plan containing the steps
            step_id: Step that depends on another
            dependency_step_id: Step that is depended upon
            
        Returns:
            True if dependency removed successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM plan_dependencies 
                WHERE plan_id = ? AND step_id = ? AND dependency_step_id = ?
            """, (plan_id, step_id, dependency_step_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Removed dependency: {step_id} -> {dependency_step_id}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error removing dependency: {e}")
            return False
    
    def get_step_dependencies(self, plan_id: str, step_id: str) -> List[Dict[str, Any]]:
        """
        Get all dependencies for a specific step.
        
        Args:
            plan_id: Plan containing the step
            step_id: Step to get dependencies for
            
        Returns:
            List of dependency records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pd.*, ps.subgoal as dependency_subgoal
                FROM plan_dependencies pd
                LEFT JOIN plan_steps ps ON pd.dependency_step_id = ps.step_id
                WHERE pd.plan_id = ? AND pd.step_id = ?
            """, (plan_id, step_id))
            
            dependencies = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return dependencies
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting step dependencies: {e}")
            return []
    
    def get_step_dependents(self, plan_id: str, step_id: str) -> List[Dict[str, Any]]:
        """
        Get all steps that depend on a specific step.
        
        Args:
            plan_id: Plan containing the step
            step_id: Step to get dependents for
            
        Returns:
            List of dependent step records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pd.*, ps.subgoal as dependent_subgoal
                FROM plan_dependencies pd
                LEFT JOIN plan_steps ps ON pd.step_id = ps.step_id
                WHERE pd.plan_id = ? AND pd.dependency_step_id = ?
            """, (plan_id, step_id))
            
            dependents = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return dependents
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting step dependents: {e}")
            return []
    
    def chain_plans(self, source_plan_id: str, target_plan_id: str, 
                   chain_type: str = "sequential", trigger_condition: Optional[str] = None) -> bool:
        """
        Chain two plans together.
        
        Args:
            source_plan_id: Plan that triggers the chain
            target_plan_id: Plan that follows in the chain
            chain_type: Type of chaining (sequential, conditional, parallel)
            trigger_condition: Optional JSON condition for activation
            
        Returns:
            True if chaining successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO plan_chains (
                    source_plan_id, target_plan_id, chain_type, trigger_condition, priority
                ) VALUES (?, ?, ?, ?, ?)
            """, (source_plan_id, target_plan_id, chain_type, trigger_condition, 1))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Chained plans: {source_plan_id} -> {target_plan_id} ({chain_type})")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error chaining plans: {e}")
            return False
    
    def get_plan_chains(self, plan_id: str, direction: str = "outgoing") -> List[Dict[str, Any]]:
        """
        Get plan chains for a specific plan.
        
        Args:
            plan_id: Plan to get chains for
            direction: "outgoing" for plans this leads to, "incoming" for plans that lead to this
            
        Returns:
            List of plan chain records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if direction == "outgoing":
                cursor.execute("""
                    SELECT pc.*, p.goal_text as target_goal
                    FROM plan_chains pc
                    LEFT JOIN plans p ON pc.target_plan_id = p.plan_id
                    WHERE pc.source_plan_id = ?
                    ORDER BY pc.priority
                """, (plan_id,))
            else:  # incoming
                cursor.execute("""
                    SELECT pc.*, p.goal_text as source_goal
                    FROM plan_chains pc
                    LEFT JOIN plans p ON pc.source_plan_id = p.plan_id
                    WHERE pc.target_plan_id = ?
                    ORDER BY pc.priority
                """, (plan_id,))
            
            chains = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return chains
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting plan chains: {e}")
            return []
    
    def add_checkpoint(self, plan_id: str, step_id: str, checkpoint_name: str,
                      checkpoint_type: str = "progress", validation_criteria: Optional[str] = None) -> bool:
        """
        Add a checkpoint to a plan step.
        
        Args:
            plan_id: Plan containing the step
            step_id: Step to add checkpoint to
            checkpoint_name: Name of the checkpoint
            checkpoint_type: Type of checkpoint (progress, validation, milestone)
            validation_criteria: Optional JSON validation criteria
            
        Returns:
            True if checkpoint added successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO plan_checkpoints (
                    plan_id, step_id, checkpoint_name, checkpoint_type, validation_criteria
                ) VALUES (?, ?, ?, ?, ?)
            """, (plan_id, step_id, checkpoint_name, checkpoint_type, validation_criteria))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Added checkpoint '{checkpoint_name}' to step {step_id}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error adding checkpoint: {e}")
            return False
    
    def reach_checkpoint(self, plan_id: str, step_id: str, checkpoint_name: str) -> bool:
        """
        Mark a checkpoint as reached.
        
        Args:
            plan_id: Plan containing the checkpoint
            step_id: Step containing the checkpoint
            checkpoint_name: Name of the checkpoint reached
            
        Returns:
            True if checkpoint marked successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE plan_checkpoints 
                SET reached_at = ?
                WHERE plan_id = ? AND step_id = ? AND checkpoint_name = ?
            """, (datetime.now().isoformat(), plan_id, step_id, checkpoint_name))
            
            conn.commit()
            conn.close()
            
            logging.info(f"[PlanMemory] Reached checkpoint '{checkpoint_name}' in step {step_id}")
            return True
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error marking checkpoint reached: {e}")
            return False
    
    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """
        Get comprehensive progress information for a plan.
        
        Args:
            plan_id: Plan to get progress for
            
        Returns:
            Progress information with checkpoints, dependencies, and status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get plan and steps
            cursor.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,))
            plan = dict(cursor.fetchone() or {})
            
            cursor.execute("SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_order", (plan_id,))
            steps = [dict(row) for row in cursor.fetchall()]
            
            # Get checkpoints
            cursor.execute("""
                SELECT * FROM plan_checkpoints 
                WHERE plan_id = ? 
                ORDER BY checkpoint_name
            """, (plan_id,))
            checkpoints = [dict(row) for row in cursor.fetchall()]
            
            # Calculate progress metrics
            total_steps = len(steps)
            completed_steps = len([s for s in steps if s['status'] == 'completed'])
            failed_steps = len([s for s in steps if s['status'] == 'failed'])
            in_progress_steps = len([s for s in steps if s['status'] == 'in_progress'])
            
            reached_checkpoints = len([c for c in checkpoints if c['reached_at']])
            total_checkpoints = len(checkpoints)
            
            progress = {
                "plan_id": plan_id,
                "goal_text": plan.get("goal_text", ""),
                "overall_status": plan.get("status", "unknown"),
                "progress_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
                "steps": {
                    "total": total_steps,
                    "completed": completed_steps,
                    "failed": failed_steps,
                    "in_progress": in_progress_steps,
                    "pending": total_steps - completed_steps - failed_steps - in_progress_steps
                },
                "checkpoints": {
                    "total": total_checkpoints,
                    "reached": reached_checkpoints,
                    "remaining": total_checkpoints - reached_checkpoints
                },
                "details": {
                    "steps": steps,
                    "checkpoints": checkpoints
                }
            }
            
            conn.close()
            return progress
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error getting plan progress: {e}")
            return {"error": str(e)}
    
    def analyze_plan_dependencies(self, plan_id: str) -> Dict[str, Any]:
        """
        Analyze the dependency structure of a plan for potential issues.
        
        Args:
            plan_id: Plan to analyze
            
        Returns:
            Analysis results with dependency issues and recommendations
        """
        try:
            dag = self.get_plan_dag(plan_id)
            if "error" in dag:
                return dag
            
            analysis = {
                "plan_id": plan_id,
                "dependency_issues": [],
                "recommendations": [],
                "metrics": {}
            }
            
            # Check for circular dependencies
            circular_deps = self._detect_circular_dependencies(dag)
            if circular_deps:
                analysis["dependency_issues"].append({
                    "type": "circular_dependency",
                    "description": "Circular dependencies detected",
                    "affected_steps": circular_deps
                })
                analysis["recommendations"].append("Break circular dependencies by reordering steps or adding conditions")
            
            # Check for orphaned steps (no dependencies and no dependents)
            orphaned_steps = self._find_orphaned_steps(dag)
            if orphaned_steps:
                analysis["dependency_issues"].append({
                    "type": "orphaned_steps",
                    "description": "Steps with no dependencies or dependents",
                    "affected_steps": orphaned_steps
                })
                analysis["recommendations"].append("Review orphaned steps for proper integration into the plan flow")
            
            # Calculate dependency metrics
            analysis["metrics"] = {
                "total_dependencies": len(dag["dependencies"]),
                "average_dependencies_per_step": len(dag["dependencies"]) / len(dag["nodes"]) if dag["nodes"] else 0,
                "dependency_depth": self._calculate_dependency_depth(dag),
                "parallelizable_steps": len(self._find_parallelizable_steps(dag))
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"[PlanMemory] Error analyzing plan dependencies: {e}")
            return {"error": str(e)}
    
    def _detect_circular_dependencies(self, dag: Dict[str, Any]) -> List[str]:
        """Detect circular dependencies in a DAG."""
        # Simplified circular dependency detection
        # In a real implementation, this would use graph algorithms
        visited = set()
        recursion_stack = set()
        
        def has_cycle(node_id, adj_list):
            if node_id in recursion_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            recursion_stack.add(node_id)
            
            for neighbor in adj_list.get(node_id, []):
                if has_cycle(neighbor, adj_list):
                    return True
            
            recursion_stack.remove(node_id)
            return False
        
        # Build adjacency list from dependencies
        adj_list = {}
        for dep in dag["dependencies"]:
            source = dep["dependency_step_id"]
            target = dep["step_id"]
            if source not in adj_list:
                adj_list[source] = []
            adj_list[source].append(target)
        
        # Check for cycles
        circular_steps = []
        for node in dag["nodes"]:
            if has_cycle(node["id"], adj_list):
                circular_steps.append(node["id"])
        
        return circular_steps
    
    def _find_orphaned_steps(self, dag: Dict[str, Any]) -> List[str]:
        """Find steps that have no dependencies or dependents."""
        step_ids = {node["id"] for node in dag["nodes"]}
        connected_steps = set()
        
        for dep in dag["dependencies"]:
            connected_steps.add(dep["step_id"])
            connected_steps.add(dep["dependency_step_id"])
        
        for edge in dag["edges"]:
            connected_steps.add(edge["from"])
            connected_steps.add(edge["to"])
        
        return list(step_ids - connected_steps)
    
    def _calculate_dependency_depth(self, dag: Dict[str, Any]) -> int:
        """Calculate the maximum dependency depth in the DAG."""
        # Simplified depth calculation
        # Would use topological sorting in a real implementation
        return max(len(node.get("prerequisites", [])) for node in dag["nodes"]) if dag["nodes"] else 0
    
    def _find_parallelizable_steps(self, dag: Dict[str, Any]) -> List[str]:
        """Find steps that can be executed in parallel."""
        # Steps with no dependencies can potentially run in parallel
        parallelizable = []
        
        for node in dag["nodes"]:
            if not node.get("prerequisites"):
                parallelizable.append(node["id"])
        
        return parallelizable