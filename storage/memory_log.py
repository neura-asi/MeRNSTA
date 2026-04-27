#!/usr/bin/env python3
"""
Universal memory log with centralized configuration
"""

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.spatial.distance import cosine

from config.settings import (
    CONFIDENCE_THRESHOLDS,
    CONFIDENCE_ICONS,
    DATABASE_CONFIG,
    DEFAULT_PERSONALITY,
    DEFAULT_VALUES,
    FACT_EXTRACTION_PATTERNS,
    FALLBACK_EXTRACTION_PATTERNS,
    PERSONALITY_PROFILES,
    VOLATILITY_THRESHOLDS,
    SIMILARITY_THRESHOLD,
    CLIP_MODEL,
    WHISPER_MODEL,
    MULTIMODAL_SIMILARITY_THRESHOLD,
    ollama_host,
    embedding_model,
    CROSS_SESSION_SEARCH_ENABLED,
)
from scripts.embedder import embed
from storage.db_utils import get_conn, with_retry, get_connection_pool, ConnectionConfig
from storage.errors import DatabaseError, safe_db_operation
from storage.formatters import format_fact_line
from storage.memory_utils import TripletFact
from storage.spacy_extractor import extract_triplets

# Import enhanced components if available
try:
    from storage.enhanced_memory_system import EnhancedMemorySystem
    from storage.enhanced_memory_model import EnhancedTripletFact
    ENHANCED_MODE = True
except ImportError:
    ENHANCED_MODE = False

Fact = TripletFact


@dataclass
class MemoryEntry:
    """Represents a memory entry in the universal log"""

    id: int
    timestamp: str
    role: str  # 'user' or 'assistant'
    content: str
    embedding: Optional[bytes] = None
    tags: Optional[List[str]] = None


class MemoryLog:
    """
    Persistent memory architecture with universal logging and SPO triplet extraction.
    """

    def __init__(self, db_path: str = DATABASE_CONFIG["default_path"]):
        self.db_path = db_path

        # Initialize connection pool for this database
        config = ConnectionConfig(db_path=db_path)
        # For test isolation, reset pool only if path is different
        self._connection_pool = get_connection_pool(config)

        self.init_database()

        # One-time database integrity check (only after tables are created)
        try:
            self._clean_malformed_facts()
        except Exception as e:
            logging.warning(f"Database cleanup skipped (tables may not exist yet): {e}")

        from storage.spacy_extractor import OllamaEmbedder
        self.embedder = OllamaEmbedder(ollama_host, embedding_model)
        
        # Initialize configurable vectorizer and hybrid memory
        try:
            from vector_memory.config import get_configured_vectorizer
            from vector_memory import HybridVectorMemory
            
            self.vectorizer = get_configured_vectorizer()
            self.hybrid_memory = HybridVectorMemory()
            
            logging.info("✅ Configurable vectorizer and hybrid memory initialized")
            logging.info(f"🧠 Hybrid mode: {self.hybrid_memory.hybrid_mode}")
            logging.info(f"🔄 Strategy: {self.hybrid_memory.hybrid_strategy}")
            
        except Exception as e:
            logging.warning(f"⚠️ Hybrid memory initialization failed, using fallback: {e}")
            # Fallback to existing embedder
            def fallback_vectorizer(text: str) -> list:
                from scripts.embedder import embed
                import numpy as np
                embedding = embed(text)
                if isinstance(embedding, np.ndarray):
                    return embedding.tolist()
                return embedding if isinstance(embedding, list) else [0.0] * 384
            self.vectorizer = fallback_vectorizer
            self.hybrid_memory = None
        
        # Initialize enhanced memory system if available
        if ENHANCED_MODE:
            self.enhanced_memory = EnhancedMemorySystem(
                db_path=db_path,
                ollama_host=ollama_host,
                embedding_model=embedding_model
            )
            logging.info("✅ Enhanced memory system initialized")

    def __del__(self):
        """Cleanup when MemoryLog is destroyed"""
        try:
            if hasattr(self, "_connection_pool"):
                # Note: Don't shutdown the pool here as it's shared
                pass
        except Exception as e:
            logging.warning(f"Error during MemoryLog cleanup: {e}")

    def shutdown(self):
        """Shutdown the memory log and cleanup resources"""
        try:
            if hasattr(self, "_connection_pool"):
                # Note: Don't shutdown the pool here as it's shared
                pass
            logging.info("MemoryLog shutdown completed")
        except Exception as e:
            logging.error(f"Error during MemoryLog shutdown: {e}")

    def init_database(self):
        """Initialize the memory database with all required tables (SPO triplets)"""
        import sqlite3

        # For in-memory databases, use the connection pool to ensure same database instance
        if self.db_path == ":memory:":
            with self._connection_pool.get_connection() as conn:
                c = conn.cursor()
                
                # Enable WAL mode for better concurrency (skip for in-memory)
                c.execute("PRAGMA synchronous=NORMAL;")
                c.execute("PRAGMA busy_timeout=30000;")
                c.execute("PRAGMA foreign_keys=ON;")
                
                self._create_tables(c, conn)
        else:
            # Use pooled connection for initialization to avoid visibility issues
            try:
                with self._connection_pool.get_connection() as conn:
                    c = conn.cursor()
                    # PRAGMAs are already configured in pooled connections, but ensure basic ones
                    try:
                        c.execute("PRAGMA foreign_keys=ON;")
                        c.execute("PRAGMA busy_timeout=30000;")
                    except Exception:
                        pass
                    self._create_tables(c, conn)
            except Exception as e:
                # Fallback to direct connection if pool fails (e.g., disk I/O edge cases under tests)
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(self.db_path, timeout=30.0)
                c = conn.cursor()
                try:
                    c.execute("PRAGMA journal_mode=WAL;")
                    c.execute("PRAGMA synchronous=NORMAL;")
                    c.execute("PRAGMA busy_timeout=30000;")
                    c.execute("PRAGMA foreign_keys=ON;")
                except Exception:
                    pass
                self._create_tables(c, conn)
                conn.close()

        # Verify tables were created by checking with connection pool
        if hasattr(self, '_connection_pool'):
            with self._connection_pool.get_connection() as conn:
                # Verify episodes table exists
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'").fetchone()
                if not result:
                    # For in-memory databases, table creation and verification use the same connection
                    # so this should not fail. If it does, it's a real error.
                    logging.error("Episodes table not found after creation")
                    if self.db_path == ":memory:":
                        # Show what tables do exist for debugging
                        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                        logging.error(f"Available tables in memory DB: {[t[0] for t in tables]}")
                    raise RuntimeError("Failed to create episodes table")

        # One-time database integrity check (only after tables are created)
        try:
            self._clean_malformed_facts()
        except Exception as e:
            logging.warning(f"Database cleanup skipped (tables may not exist yet): {e}")

        from storage.spacy_extractor import OllamaEmbedder
        self.embedder = OllamaEmbedder(ollama_host, embedding_model)

    def _create_tables(self, c, conn):
        """Create all database tables"""
        # Facts table
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                source_message_id INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                frequency INTEGER DEFAULT 1,
                contradiction_score REAL DEFAULT 0.0,
                volatility_score REAL DEFAULT 0.0,
                confidence REAL DEFAULT 1.0,
                last_reinforced TEXT DEFAULT CURRENT_TIMESTAMP,
                episode_id INTEGER DEFAULT 1,
                emotion_score REAL DEFAULT NULL,
                context TEXT DEFAULT NULL,
                media_type TEXT DEFAULT 'text',
                media_data BLOB,
                embedding BLOB,
                user_profile_id TEXT,
                session_id TEXT,
                subject_cluster_id TEXT
            )
        """)
        
        # Episodes table
        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                end_time TEXT,
                subject_count INTEGER DEFAULT 0,
                fact_count INTEGER DEFAULT 0,
                summary TEXT
            )
        """)
        
        # Clusters table
        c.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                embedding BLOB,
                fact_ids TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                trust_score REAL DEFAULT 1.0,
                cluster_size INTEGER DEFAULT 1
            )
        """)
        
        # Drift events table
        c.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                fact_id INTEGER,
                drift_value REAL,
                resolution_action TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trust scores table
        c.execute("""
            CREATE TABLE IF NOT EXISTS trust_scores (
                subject TEXT PRIMARY KEY,
                trust_score REAL,
                fact_count INTEGER DEFAULT 0,
                contradiction_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Universal memory log
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                role TEXT,
                content TEXT,
                embedding BLOB,
                tags TEXT
            )
        """)
        
        # Fact history table for tracking changes
        c.execute("""
            CREATE TABLE IF NOT EXISTS fact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                old_value TEXT,
                new_value TEXT,
                change_type TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fact_id) REFERENCES facts (id)
            )
        """)
        
        # Contradictions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                fact_a_text TEXT,
                fact_b_text TEXT,
                confidence REAL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fact_a_id) REFERENCES facts (id),
                FOREIGN KEY (fact_b_id) REFERENCES facts (id)
            )
        """)
        
        # Summaries table
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                summary_text TEXT,
                fact_count INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
                # Create initial episode if none exists
        c.execute("""
            INSERT OR IGNORE INTO episodes (id, start_time, subject_count, fact_count, summary)
            VALUES (1, CURRENT_TIMESTAMP, 0, 0, 'Initial episode')
        """)

        conn.commit()
                
    def _clean_malformed_facts(self):
        """Clean any malformed facts from the database on startup"""
        try:
            with self._connection_pool.get_connection() as conn:
                # Check if facts table exists first
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
                )
                if not cursor.fetchone():
                    logging.info("Facts table doesn't exist yet, skipping cleanup")
                    return

                cursor.execute(
                    "SELECT COUNT(*) FROM facts WHERE subject IS NULL OR predicate IS NULL OR object IS NULL"
                )
                malformed_count = cursor.fetchone()[0]

                if malformed_count > 0:
                    print(
                        f"🧹 Cleaning {malformed_count} malformed facts from database..."
                    )
                    cursor.execute(
                        "DELETE FROM facts WHERE subject IS NULL OR predicate IS NULL OR object IS NULL"
                    )
                    conn.commit()
                    print(
                        f"{CONFIDENCE_ICONS['success']} Database cleaned - removed {malformed_count} malformed facts"
                    )
        except Exception as e:
            logging.warning(f"Database cleanup skipped: {e}")

    @safe_db_operation
    def log_memory(self, role: str, content: str, tags: List[str] = None) -> int:
        for attempt in range(3):
            try:
                # Generate embedding for semantic search using configurable vectorizer
                try:
                    vector = self.vectorizer(content)
                    # Convert to numpy array for compatibility
                    embedding = np.array(vector, dtype=np.float32)
                    if np.all(embedding == 0):
                        print(f"Warning: Vectorizer returned zero vector for content '{content}' in log_memory.")
                except Exception as e:
                    print(f"Warning: Vectorizer failed for content '{content}': {e}, using fallback")
                    # Fallback to original embedder
                    from scripts.embedder import embed
                    embedding = embed(content)
                    if np.all(embedding == 0):
                        print(f"Warning: Fallback embedding also failed for content '{content}' in log_memory.")
                
                embedding = embedding.tobytes()

                # Convert tags to JSON string
                tags_json = json.dumps(tags) if tags else None

                try:
                    with self._connection_pool.get_connection() as conn:
                        cursor = conn.execute(
                            "INSERT INTO memory (role, content, embedding, tags) VALUES (?, ?, ?, ?)",
                            (role, content, embedding, tags_json),
                        )
                        conn.commit()
                        return cursor.lastrowid
                except Exception as e:
                    logging.error(f"[DB ERROR] log_memory failed: {e}", exc_info=True)
                    raise DatabaseError(f"log_memory failed: {e}") from e
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < 2:
                    time.sleep(0.1)
                    continue
                raise

    @safe_db_operation
    def fetch_recent_context(
        self, n: int = DEFAULT_VALUES["recent_context_limit"]
    ) -> List[MemoryEntry]:
        """
        Fetch the most recent memory entries for context

        Args:
            n: Number of recent entries to fetch

        Returns:
            List of MemoryEntry objects
        """
        try:
            with self._connection_pool.get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, role, content, tags FROM memory ORDER BY id DESC LIMIT ?",
                    (n,),
                ).fetchall()

                entries = []
                for row in reversed(rows):  # Reverse to get chronological order
                    id_val, timestamp, role, content, tags_json = row
                    tags = json.loads(tags_json) if tags_json else None
                    entries.append(
                        MemoryEntry(id_val, timestamp, role, content, tags=tags)
                    )

                return entries
        except Exception as e:
            logging.error(f"[DB ERROR] fetch_recent_context failed: {e}", exc_info=True)
            raise DatabaseError(f"fetch_recent_context failed: {e}") from e

    @safe_db_operation
    def fetch_semantic_context(
        self, query: str, n: int = DEFAULT_VALUES["semantic_context_limit"]
    ) -> List[MemoryEntry]:
        """
        Fetch memory entries semantically similar to the query

        Args:
            query: The search query
            n: Number of similar entries to fetch

        Returns:
            List of MemoryEntry objects ordered by similarity
        """
        query_embedding = embed(query)
        if np.all(query_embedding == 0):
            print(
                f"Warning: Embedding failed for query '{query}' in fetch_semantic_context."
            )

        try:
            with self._connection_pool.get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, role, content, embedding, tags FROM memory"
                ).fetchall()

                similarities = []
                for row in rows:
                    id_val, timestamp, role, content, embedding_blob, tags_json = row
                    if embedding_blob:
                        memory_embedding = np.frombuffer(
                            embedding_blob, dtype=np.float32
                        )
                        similarity = np.dot(query_embedding, memory_embedding)
                        similarities.append(
                            (
                                similarity,
                                MemoryEntry(
                                    id_val,
                                    timestamp,
                                    role,
                                    content,
                                    embedding_blob,
                                    json.loads(tags_json) if tags_json else None,
                                ),
                            )
                        )

                # Sort by similarity and return top n
                similarities.sort(key=lambda x: x[0], reverse=True)
                return [entry for _, entry in similarities[:n]]
        except Exception as e:
            logging.error(
                f"[DB ERROR] fetch_semantic_context failed: {e}", exc_info=True
            )
            raise DatabaseError(f"fetch_semantic_context failed: {e}") from e

    @safe_db_operation
    def resolve_pronouns(self, user_input: str) -> str:
        """
        Resolve pronouns and coreferences in user input using recent memory context.
        Replaces "them", "it", "that", etc. with actual entity names from recent memory.
        """
        # Get recent memory context to find potential referents
        recent_entries = self.fetch_recent_context(10)
        recent_text = " ".join([entry.content for entry in recent_entries])

        # Common pronouns to resolve
        pronouns = {
            "them": [],
            "it": [],
            "that": [],
            "this": [],
            "those": [],
            "these": [],
            "him": [],
            "her": [],
            "he": [],
            "she": [],
        }

        # Extract potential entities from recent memory
        potential_entities = set()

        # Look for noun phrases in recent memory
        import re

        # Simple pattern to find potential entities (words that could be subjects/objects)
        noun_patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",  # Capitalized phrases
            r"\b([a-z]+(?:\s+[a-z]+)*)\b",  # Lowercase phrases (common nouns)
        ]

        for pattern in noun_patterns:
            matches = re.findall(pattern, recent_text)
            for match in matches:
                if len(match.split()) <= 3:  # Limit to reasonable entity length
                    potential_entities.add(match.lower())

        # Also extract from existing facts
        existing_facts = self.get_all_facts()
        for fact in existing_facts:
            if fact.subject:
                potential_entities.add(fact.subject.lower())
            if fact.object:
                potential_entities.add(fact.object.lower())

        # Filter out common words that aren't entities
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "must",
            "shall",
        }
        potential_entities = {
            entity for entity in potential_entities if entity not in common_words
        }

        # Try LLM-based resolution for better accuracy
        try:
            import requests

            prompt = f"""Resolve pronouns in this text by replacing them with the most likely referent from the recent context.

Recent context: {recent_text[:500]}

Available entities: {', '.join(list(potential_entities)[:20])}

Text to resolve: "{user_input}"

Replace pronouns like "them", "it", "that", "this", "those", "these" with the most likely entity from the context. If unsure, keep the original pronoun.

Resolved text:"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=10,
            )
            response.raise_for_status()

            resolved_text = response.json()["response"].strip()
            # Remove quotes if LLM added them
            if resolved_text.startswith('"') and resolved_text.endswith('"'):
                resolved_text = resolved_text[1:-1]

            return resolved_text

        except Exception as e:
            print(f"Warning: LLM pronoun resolution failed: {e}")
            # Fallback to simple rule-based resolution
            return self._simple_pronoun_resolution(user_input, potential_entities)

    def _simple_pronoun_resolution(self, text: str, entities: set) -> str:
        """
        Simple rule-based pronoun resolution as fallback.
        """
        resolved_text = text

        # Get the most recent entity mentioned
        recent_entities = list(entities)[:5]  # Take top 5 most recent

        # Simple replacements based on context
        if "them" in text.lower() and recent_entities:
            resolved_text = re.sub(
                r"\bthem\b", recent_entities[0], resolved_text, flags=re.IGNORECASE
            )
        if "it" in text.lower() and recent_entities:
            resolved_text = re.sub(
                r"\bit\b", recent_entities[0], resolved_text, flags=re.IGNORECASE
            )
        if "that" in text.lower() and recent_entities:
            resolved_text = re.sub(
                r"\bthat\b", recent_entities[0], resolved_text, flags=re.IGNORECASE
            )

        return resolved_text

    @safe_db_operation
    def extract_triplets(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        Extract subject-predicate-object triplets from text using spaCy dependency parsing.
        Replaces hardcoded regex patterns with data-driven linguistic analysis.
        """
        # PRE-PROCESS: Handle compound statements with conjunctions
        original_text = text
        compound_clauses = self._split_compound_statements(text)
        
        all_triplets = []
        last_subject = None  # Track subject from previous clause for compound statements
        
        for i, clause in enumerate(compound_clauses):
            # Check if this clause starts with a verb (no subject)
            first_word = clause.split()[0].lower() if clause.split() else ""
            
            # Use linguistic analysis to detect if clause starts with a verb
            starts_with_verb = False
            try:
                from storage.spacy_extractor import nlp
                if nlp and clause:
                    doc = nlp(clause)
                    if doc and len(doc) > 0:
                        # Check if first token is a verb
                        if doc[0].pos_ == 'VERB':
                            starts_with_verb = True
            except:
                # Fallback: check if it's a common verb form
                # This is minimal and will be replaced by proper NLP
                if first_word.endswith(('s', 'ed', 'ing')) and len(first_word) > 3:
                    starts_with_verb = True
            
            # If clause starts with a verb and we have a previous subject, prepend it
            if starts_with_verb and last_subject and i > 0:
                clause = f"{last_subject} {clause}"
                logging.info(f"[Compound Statement] Added subject '{last_subject}' to clause: {clause}")
            try:
                from storage.spacy_extractor import extractor
                if extractor:
                    spacy_triplets = extractor.extract_triplets(clause)
                    if spacy_triplets:
                        legacy_triplets = extractor.convert_to_legacy_format(spacy_triplets)
                        cleaned_triplets = []
                        for subject, predicate, object_, confidence in legacy_triplets:
                            cleaned_subject = self._clean_text(subject)
                            cleaned_predicate = self._clean_text(predicate)
                            cleaned_object = self._clean_text(object_)
                            # Canonicalize subject for user preferences
                            canonical_subject = self._canonicalize_subject(cleaned_subject, cleaned_predicate, cleaned_object)
                            canonical_predicate = self._canonicalize_predicate(cleaned_predicate)
                            logging.info(f"[Triplet Extraction] Canonical subject: '{canonical_subject}', predicate: '{canonical_predicate}', object: '{cleaned_object}' (from '{cleaned_subject}', '{cleaned_predicate}', '{cleaned_object}')")
                            if canonical_subject and cleaned_object:
                                cleaned_triplets.append((canonical_subject, canonical_predicate, cleaned_object, confidence))
                        all_triplets.extend(cleaned_triplets)
                        # Remember subject for next clause
                        if cleaned_triplets and cleaned_triplets[0][0]:
                            last_subject = cleaned_triplets[0][0]
                    else:
                        logging.info(f"No triplets extracted by spaCy for clause '{clause}', falling back to regex patterns")
                        regex_triplets = self._extract_triplets_regex(clause)
                        all_triplets.extend(regex_triplets)
                        # Remember subject for next clause
                        if regex_triplets and regex_triplets[0][0]:
                            last_subject = regex_triplets[0][0]
            except ImportError:
                logging.warning(f"spaCy extractor not available for clause '{clause}', using regex patterns")
                regex_triplets = self._extract_triplets_regex(clause)
                all_triplets.extend(regex_triplets)
                # Remember subject for next clause
                if regex_triplets and regex_triplets[0][0]:
                    last_subject = regex_triplets[0][0]
            except Exception as e:
                logging.warning(f"Error in spaCy extraction for clause '{clause}': {e}, falling back to regex patterns")
                regex_triplets = self._extract_triplets_regex(clause)
                all_triplets.extend(regex_triplets)
                # Remember subject for next clause
                if regex_triplets and regex_triplets[0][0]:
                    last_subject = regex_triplets[0][0]
        
        # If no triplets found from clauses, try original text as fallback
        if not all_triplets:
            triplets = self._extract_triplets_regex(original_text)
            normalized_triplets = []
            for subject, predicate, object_, confidence in triplets:
                canonical_subject = self._canonicalize_subject(subject, predicate, object_)
                canonical_predicate = self._canonicalize_predicate(predicate)
                logging.info(f"[Triplet Extraction] Canonical subject: '{canonical_subject}', predicate: '{canonical_predicate}', object: '{object_}' (from '{subject}', '{predicate}', '{object_}')")
                normalized_triplets.append((canonical_subject, canonical_predicate, object_, confidence))
            return normalized_triplets
        
        # Normalize compound statement triplets
        normalized_triplets = []
        for subject, predicate, object_, confidence in all_triplets:
            canonical_subject = self._canonicalize_subject(subject, predicate, object_)
            canonical_predicate = self._canonicalize_predicate(predicate)
            logging.info(f"[Compound Triplet Extraction] Canonical subject: '{canonical_subject}', predicate: '{canonical_predicate}', object: '{object_}' (from '{subject}', '{predicate}', '{object_}')")
            normalized_triplets.append((canonical_subject, canonical_predicate, object_, confidence))
        
        return normalized_triplets

    def _split_compound_statements(self, text: str) -> List[str]:
        """
        Split compound statements on conjunctions like 'but', 'and', 'however', etc.
        Returns list of individual clauses that can be processed separately.
        """
        import re
        
        if not text:
            return []
        
        # Special handling for temporal update patterns
        # Pattern: "X used to be Y but now it's Z"
        temporal_update_pattern = r'(.+?)\s+used\s+to\s+be\s+(.+?)\s+but\s+now\s+(?:it\'s|its|it\s+is)\s+(.+)'
        temporal_match = re.match(temporal_update_pattern, text, re.IGNORECASE)
        
        if temporal_match:
            # This is a temporal update - extract as two facts:
            # 1. Past state: "X was Y" (with low confidence or marked as outdated)
            # 2. Current state: "X is Z"
            subject_part = temporal_match.group(1).strip()
            old_value = temporal_match.group(2).strip()
            new_value = temporal_match.group(3).strip()
            
            # Create two statements
            past_statement = f"{subject_part} was {old_value} in the past"
            current_statement = f"{subject_part} is {new_value}"
            
            logging.info(f"[Temporal Update] Detected temporal pattern: '{text}'")
            logging.info(f"[Temporal Update] Past: '{past_statement}', Current: '{current_statement}'")
            
            return [current_statement]  # Only return current state as the valid fact
        
        # Check for other temporal patterns like "isn't X anymore"
        negative_temporal_pattern = r'(.+?)\s+(?:isn\'t|is\s+not|aren\'t|are\s+not)\s+(.+?)\s+anymore'
        neg_temporal_match = re.match(negative_temporal_pattern, text, re.IGNORECASE)
        
        if neg_temporal_match:
            subject = neg_temporal_match.group(1).strip()
            object_part = neg_temporal_match.group(2).strip()
            
            # Create a negative statement
            statement = f"{subject} no longer has {object_part}"
            logging.info(f"[Negative Temporal] Detected pattern: '{text}' -> '{statement}'")
            return [statement]
        
        # Standard conjunction splitting for other cases
        conjunctions = [
            r'\s+but\s+',
            r'\s+however\s+',
            r'\s+although\s+',
            r'\s+though\s+',
            r'\s+while\s+',
            r'[,;]\s*and\s+',  # "and" after comma/semicolon (not simple lists)
        ]
        
        # Try to split on conjunctions
        clauses = [text]
        
        for conjunction_pattern in conjunctions:
            new_clauses = []
            for clause in clauses:
                # Skip splitting if this looks like a temporal update
                if 'used to be' in clause.lower() and 'but now' in clause.lower():
                    new_clauses.append(clause)
                    continue
                    
                # Split on this conjunction
                parts = re.split(conjunction_pattern, clause, flags=re.IGNORECASE)
                new_clauses.extend([part.strip() for part in parts if part.strip()])
            clauses = new_clauses
        
        # Filter out very short clauses (likely artifacts)
        meaningful_clauses = [clause for clause in clauses if len(clause.split()) >= 3]
        
        # If we filtered out everything, return original text
        if not meaningful_clauses:
            meaningful_clauses = [text]
            
        logging.info(f"[Compound Statement] Split '{text}' into {len(meaningful_clauses)} clauses: {meaningful_clauses}")
        
        return meaningful_clauses

    def _canonicalize_subject(self, subject: str, predicate: str, object_: str) -> str:
        """
        Normalize user-centric subjects to a canonical form.
        """
        if not subject:
            return ''
        
        subj = subject.lower().strip()
        
        # Handle special patterns without losing context
        if 'favorite' in subj or 'fav' in subj:
            # For "my favorite X", return "user favorite X"
            # Be careful to only replace whole words
            import re
            subj_normalized = re.sub(r'\bmy\b', 'user', subj)
            subj_normalized = re.sub(r'\bi\b', 'user', subj_normalized)
            logging.info(f"[CANONICALIZE] favorite pattern: '{subj}' -> '{subj_normalized}'")
            return subj_normalized
        
        # Simple pronoun replacement for other cases
        if subj in ['i', 'me', 'my', 'myself']:
            return 'user'
        elif subj.startswith('my '):
            # Replace "my" with "user" but keep the rest
            return 'user ' + subj[3:]
        elif subj.startswith('i '):
            return 'user ' + subj[2:]
        
        # Return as-is for other subjects
        return subj

    def _canonicalize_predicate(self, predicate: str) -> str:
        """
        Normalize predicates for preference queries (e.g., 'is', 'are', 'like', 'prefer').
        """
        if not predicate:
            return ''
        pred = predicate.lower().strip()
        # For copula or preference verbs, just return as is for now
        return pred

    def _extract_triplets_regex(self, text: str) -> List[Tuple[str, str, str, float]]:
        """
        Original regex-based triplet extraction (fallback method)
        """
        from storage.memory_utils import normalize_predicate

        # Check if this is a question and skip extraction
        if self._is_question_regex(text):
            return []

        triplets = []

        try:
            # Test all patterns and get results
            pattern_results = self.test_patterns(text)

            # Process high-confidence matches first
            for match in pattern_results.get("high_confidence_matches", []):
                try:
                    subject = match.get("subject", "").strip()
                    predicate = match.get("predicate", "").strip()
                    object_ = match.get("object", "").strip()
                    confidence = match.get("confidence", 0.5)

                    if subject and predicate and object_:
                        # Normalize predicate to canonical form
                        normalized_predicate = normalize_predicate(predicate)

                        # Clean the extracted text
                        subject = self._clean_text(subject)
                        object_ = self._clean_text(object_)

                        if subject and object_:
                            triplets.append(
                                (subject, normalized_predicate, object_, confidence)
                            )
                except Exception as e:
                    print(f"❌ Error processing high-confidence match: {e}")
                    continue

            # Process fallback matches if no high-confidence matches
            if not triplets:
                for match in pattern_results.get("fallback_matches", []):
                    try:
                        subject = match.get("subject", "").strip()
                        predicate = match.get("predicate", "").strip()
                        object_ = match.get("object", "").strip()
                        confidence = match.get(
                            "confidence", VOLATILITY_THRESHOLDS["stable"]
                        )

                        if subject and predicate and object_:
                            # Normalize predicate to canonical form
                            normalized_predicate = normalize_predicate(predicate)

                            # Clean the extracted text
                            subject = self._clean_text(subject)
                            object_ = self._clean_text(object_)

                            if subject and object_:
                                triplets.append(
                                    (subject, normalized_predicate, object_, confidence)
                                )
                    except Exception as e:
                        print(f"❌ Error processing fallback match: {e}")
                        continue

            # Remove duplicates while preserving highest confidence
            unique_triplets = {}
            for subject, predicate, object_, confidence in triplets:
                # Defensive: skip any triplet with None values
                if subject is None or predicate is None or object_ is None:
                    print(
                        f"[WARN] Skipping triplet with None values: {subject}, {predicate}, {object_}"
                    )
                    continue
                # Defensive: ensure all values are strings
                subject = str(subject) if subject is not None else ""
                predicate = str(predicate) if predicate is not None else ""
                object_ = str(object_) if object_ is not None else ""

                key = (subject.lower(), predicate.lower(), object_.lower())
                if key not in unique_triplets or confidence > unique_triplets[key][3]:
                    unique_triplets[key] = (subject, predicate, object_, confidence)

            return list(unique_triplets.values())

        except Exception as e:
            print(f"❌ Error in extract_triplets: {e}")
            return []

    def _is_question_regex(self, text: str) -> bool:
        """
        Detect if text is a question using regex patterns (fallback for when spaCy is not available)
        
        Args:
            text: Input text
            
        Returns:
            True if text appears to be a question
        """
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

    @safe_db_operation
    def test_patterns(self, text: str) -> Dict[str, any]:
        """
        Test pattern extraction on given text and return detailed results.
        Useful for debugging and pattern development.
        """
        if not text or not isinstance(text, str):
            return {"error": "Invalid input text"}

        results = {
            "input_text": text,
            "high_confidence_matches": [],
            "fallback_matches": [],
            "total_triplets": 0,
            "pattern_stats": {},
        }

        # Test high-confidence patterns
        for i, (pattern, confidence) in enumerate(FACT_EXTRACTION_PATTERNS):
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    pattern_results = []
                    for match in matches:
                        # Use groupdict() for named groups - returns empty string if group missing
                        group_dict = match.groupdict()
                        subject = group_dict.get("subject", "").strip()
                        predicate = group_dict.get("predicate", "").strip()
                        obj = group_dict.get("object", "").strip()

                        if subject and predicate and obj:
                            subject = self._clean_text(subject)
                            predicate = self._clean_text(predicate)
                            obj = self._clean_text(obj)

                            if subject and predicate and obj:
                                pattern_results.append(
                                    {
                                        "subject": subject,
                                        "predicate": predicate,
                                        "object": obj,
                                        "confidence": confidence,
                                        "full_match": match.group(0),
                                    }
                                )

                    if pattern_results:
                        results["high_confidence_matches"].extend(pattern_results)
                        results["pattern_stats"][f"pattern_{i}"] = {
                            "pattern": pattern,
                            "confidence": confidence,
                            "matches": len(pattern_results),
                        }
            except Exception as e:
                results["pattern_stats"][f"pattern_{i}"] = {
                    "pattern": pattern,
                    "confidence": confidence,
                    "error": str(e),
                }

        # Test fallback patterns if no high-confidence matches
        if not results["high_confidence_matches"]:
            for i, (pattern, confidence) in enumerate(FALLBACK_EXTRACTION_PATTERNS):
                try:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE))
                    if matches:
                        pattern_results = []
                        for match in matches:
                            # Extract groups by position (1, 2, 3) - check if groups exist
                            if match.lastindex and match.lastindex >= 3:
                                subject = match.group(1).strip()
                                predicate = match.group(2).strip()
                                obj = match.group(3).strip()
                            else:
                                continue  # Skip if not enough groups

                            if subject and predicate and obj:
                                subject = self._clean_text(subject)
                                predicate = self._clean_text(predicate)
                                obj = self._clean_text(obj)

                                if subject and predicate and obj:
                                    pattern_results.append(
                                        {
                                            "subject": subject,
                                            "predicate": predicate,
                                            "object": obj,
                                            "confidence": confidence,
                                            "full_match": match.group(0),
                                        }
                                    )

                        if pattern_results:
                            results["fallback_matches"].extend(pattern_results)
                            results["pattern_stats"][f"fallback_{i}"] = {
                                "pattern": pattern,
                                "confidence": confidence,
                                "matches": len(pattern_results),
                            }
                except Exception as e:
                    results["pattern_stats"][f"fallback_{i}"] = {
                        "pattern": pattern,
                        "confidence": confidence,
                        "error": str(e),
                    }

        # Calculate totals
        results["total_triplets"] = len(results["high_confidence_matches"]) + len(
            results["fallback_matches"]
        )

        return results



    @safe_db_operation
    def analyze_emotion(self, text: str) -> float:
        """
        Enhanced sentiment/emotion analyzer using VADER-like approach.
        Returns emotion_score (0–1) with better word weighting and context.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Emotion score between 0.0 and 1.0
        """
        import re

        # Enhanced emotion word dictionaries with intensity scores
        positive_words = {
            # High intensity (VOLATILITY_THRESHOLDS["high"]-1.0)
            "love": CONFIDENCE_THRESHOLDS["very_high"],
            "adore": CONFIDENCE_THRESHOLDS["very_high"],
            "ecstatic": CONFIDENCE_THRESHOLDS["very_high"],
            "passionate": CONFIDENCE_THRESHOLDS["very_high"],
            "amazing": CONFIDENCE_THRESHOLDS["high"],
            "wonderful": CONFIDENCE_THRESHOLDS["high"],
            "fantastic": CONFIDENCE_THRESHOLDS["high"],
            "incredible": CONFIDENCE_THRESHOLDS["high"],
            "excited": CONFIDENCE_THRESHOLDS["high"],
            "thrilled": CONFIDENCE_THRESHOLDS["high"],
            "delighted": CONFIDENCE_THRESHOLDS["high"],
            "joyful": CONFIDENCE_THRESHOLDS["high"],
            # Medium intensity (0.6-VOLATILITY_THRESHOLDS["high"])
            "happy": CONFIDENCE_THRESHOLDS["medium"],
            "great": CONFIDENCE_THRESHOLDS["medium"],
            "good": CONFIDENCE_THRESHOLDS["medium"],
            "nice": CONFIDENCE_THRESHOLDS["medium"],
            "enjoy": CONFIDENCE_THRESHOLDS["medium"],
            "like": CONFIDENCE_THRESHOLDS["medium"],
            "prefer": CONFIDENCE_THRESHOLDS["medium"],
            "appreciate": CONFIDENCE_THRESHOLDS["medium"],
            "pleased": CONFIDENCE_THRESHOLDS["medium"],
            "satisfied": CONFIDENCE_THRESHOLDS["medium"],
            "content": CONFIDENCE_THRESHOLDS["medium"],
            "fine": CONFIDENCE_THRESHOLDS["medium"],
            # Low intensity (VOLATILITY_THRESHOLDS["stable"]-0.6)
            "okay": CONFIDENCE_THRESHOLDS["low"],
            "alright": CONFIDENCE_THRESHOLDS["low"],
            "decent": CONFIDENCE_THRESHOLDS["low"],
            "reasonable": CONFIDENCE_THRESHOLDS["low"],
            "acceptable": CONFIDENCE_THRESHOLDS["low"],
            "adequate": CONFIDENCE_THRESHOLDS["low"],
            "sufficient": CONFIDENCE_THRESHOLDS["low"],
        }

        negative_words = {
            # High intensity (VOLATILITY_THRESHOLDS["high"]-1.0)
            "hate": CONFIDENCE_THRESHOLDS["very_high"],
            "despise": CONFIDENCE_THRESHOLDS["very_high"],
            "terrified": CONFIDENCE_THRESHOLDS["very_high"],
            "horrified": CONFIDENCE_THRESHOLDS["very_high"],
            "furious": CONFIDENCE_THRESHOLDS["high"],
            "enraged": CONFIDENCE_THRESHOLDS["high"],
            "disgusted": CONFIDENCE_THRESHOLDS["high"],
            "appalled": CONFIDENCE_THRESHOLDS["high"],
            "terrible": CONFIDENCE_THRESHOLDS["high"],
            "awful": CONFIDENCE_THRESHOLDS["high"],
            "horrible": CONFIDENCE_THRESHOLDS["high"],
            "dreadful": CONFIDENCE_THRESHOLDS["high"],
            # Medium intensity (0.6-VOLATILITY_THRESHOLDS["high"])
            "angry": CONFIDENCE_THRESHOLDS["medium"],
            "sad": CONFIDENCE_THRESHOLDS["medium"],
            "upset": CONFIDENCE_THRESHOLDS["medium"],
            "disappointed": CONFIDENCE_THRESHOLDS["medium"],
            "fear": CONFIDENCE_THRESHOLDS["medium"],
            "worried": CONFIDENCE_THRESHOLDS["medium"],
            "anxious": CONFIDENCE_THRESHOLDS["medium"],
            "stressed": CONFIDENCE_THRESHOLDS["medium"],
            "dislike": CONFIDENCE_THRESHOLDS["medium"],
            "avoid": CONFIDENCE_THRESHOLDS["medium"],
            "hate": VOLATILITY_THRESHOLDS["high"],
            "loathe": VOLATILITY_THRESHOLDS["high"],
            # Low intensity (VOLATILITY_THRESHOLDS["stable"]-0.6)
            "annoyed": CONFIDENCE_THRESHOLDS["medium"],
            "bothered": CONFIDENCE_THRESHOLDS["low"],
            "concerned": CONFIDENCE_THRESHOLDS["low"],
            "uneasy": CONFIDENCE_THRESHOLDS["low"],
            "uncomfortable": CONFIDENCE_THRESHOLDS["low"],
            "dissatisfied": CONFIDENCE_THRESHOLDS["medium"],
            "displeased": CONFIDENCE_THRESHOLDS["medium"],
        }

        # Intensifiers that amplify emotion
        intensifiers = {
            "very": 1.5,
            "really": 1.4,
            "extremely": 1.6,
            "absolutely": 1.5,
            "completely": 1.4,
            "totally": 1.3,
            "incredibly": 1.5,
            "amazingly": 1.4,
            "so": 1.2,
            "such": 1.2,
            "quite": 1.1,
            "rather": 1.1,
        }

        # Negation words that flip or reduce emotion
        negations = [
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nowhere",
            "hardly",
            "barely",
            "scarcely",
            "doesn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
        ]

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        total_score = 0.0
        word_count = len(words)

        if word_count == 0:
            return 0.1  # Neutral baseline

        for i, word in enumerate(words):
            word_score = 0.0
            multiplier = 1.0

            # Check for intensifiers
            if i > 0 and words[i - 1] in intensifiers:
                multiplier *= intensifiers[words[i - 1]]

            # Check for negations (simple approach)
            negation_context = False
            for j in range(max(0, i - 3), i):
                if words[j] in negations:
                    negation_context = True
                    break

            if negation_context:
                multiplier *= -0.5  # Reduce and potentially flip emotion

            # Check positive words
            if word in positive_words:
                word_score = positive_words[word]
            # Check negative words
            elif word in negative_words:
                word_score = negative_words[word]
                multiplier *= -1  # Make it negative

            # Apply multiplier
            word_score *= multiplier
            total_score += word_score

        # Normalize to 0-1 range
        if total_score > 0:
            # Positive emotion: map to VOLATILITY_THRESHOLDS["stable"]-1.0
            normalized_score = VOLATILITY_THRESHOLDS["stable"] + (
                min(total_score / word_count, 2.0) / 2.0
            ) * (1.0 - VOLATILITY_THRESHOLDS["stable"])
        elif total_score < 0:
            # Negative emotion: map to 0.0-VOLATILITY_THRESHOLDS["stable"]
            normalized_score = (
                VOLATILITY_THRESHOLDS["stable"]
                - (min(abs(total_score) / word_count, 2.0) / 2.0)
                * VOLATILITY_THRESHOLDS["stable"]
            )
        else:
            # Neutral: 0.1-VOLATILITY_THRESHOLDS["stable"] range
            normalized_score = (
                DEFAULT_VALUES["confidence_decay_threshold"]
                + (word_count % 3) * DEFAULT_VALUES["confidence_decay_threshold"]
            )

        return min(max(normalized_score, 0.0), 1.0)

    @safe_db_operation
    def store_triplets(self, triplets, message_id=None, session_id=None, user_profile_id=None):
        # Validate triplet format
        # Supports two formats:
        # 1. (subject, predicate, object, confidence, context, media_type, media_data)
        # 2. (subject, predicate, object, meta_dict)
        stored_ids = []
        summary_messages = []
        with self._connection_pool.get_connection() as conn:
            for triplet in triplets:
                if len(triplet) < 3:
                    continue
                subject, predicate, object_ = triplet[:3]
                
                # Handle both old and new formats
                if len(triplet) == 4 and isinstance(triplet[3], dict):
                    # New format: (subject, predicate, object, meta)
                    meta = triplet[3]
                    confidence = meta.get("confidence", 1.0)
                    context = meta.get("context", None)
                    media_type = meta.get("media_type", "text")
                    media_data = meta.get("media_data", None)
                else:
                    # Old format: (subject, predicate, object, confidence, context, media_type, media_data)
                    confidence = triplet[3] if len(triplet) > 3 else 1.0
                    context = triplet[4] if len(triplet) > 4 else None
                    media_type = triplet[5] if len(triplet) > 5 else "text"
                    media_data = triplet[6] if len(triplet) > 6 else None
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                episode_id = 1
                self._demote_older_facts_by_subject(subject, conn)
                existing_fact = self._find_similar_facts_internal(f"{subject} {predicate} {object_}", conn)
                if existing_fact:
                    self._reinforce_fact_internal(existing_fact.id, 0.1, conn)
                    stored_ids.append(existing_fact.id)
                    summary_messages.append(f"Reinforced existing fact: {subject} {predicate} {object_}")
                    continue
                cursor = conn.execute(
                    """
                    INSERT INTO facts (subject, predicate, object, confidence, timestamp, episode_id, frequency, context, media_type, media_data, session_id, user_profile_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        subject,
                        predicate,
                        object_,
                        confidence,
                        timestamp,
                        episode_id,
                        1,
                        json.dumps(context) if context else None,
                        media_type if media_type else "text",
                        media_data,
                        session_id,
                        user_profile_id,
                    ),
                )
                fact_id = cursor.lastrowid
                stored_ids.append(fact_id)
        return stored_ids, summary_messages

    def semantic_search(self, query: str, topk: int = 5, is_query: bool = False, media_type: str = "text", user_profile_id: str = None, session_id: str = None) -> list:
        """
        Semantic search for facts with hybrid memory intelligence support.
        Uses configurable vectorizers (default, HRRFormer, VecSymR) with intelligent fusion.
        """
        # Try hybrid search first if available
        if hasattr(self, 'hybrid_memory') and self.hybrid_memory and self.hybrid_memory.hybrid_mode:
            return self._hybrid_semantic_search(query, topk, is_query, media_type, user_profile_id, session_id)
        
        # Fallback to traditional semantic search
        from scripts.embedder import embed
        from config.settings import CROSS_SESSION_SEARCH_ENABLED, MULTIMODAL_SIMILARITY_THRESHOLD
        from config.environment import get_settings
        
        settings = get_settings()
        
        def get_embedding(text_input):
            from scripts.embedder import embed
            return embed(text_input)
            
        try:
            with self._connection_pool.get_connection() as conn:
                # Build SQL query with proper user_profile_id filtering
                sql_params = []
                sql_query = """
                    SELECT id, subject, predicate, object, confidence, timestamp, frequency, context, media_type, media_data
                    FROM facts
                    WHERE 1=1
                """
                
                # Filter by user_profile_id when provided
                if user_profile_id:
                    sql_query += " AND user_profile_id = ?"
                    sql_params.append(user_profile_id)
                
                if session_id:
                    sql_query += " AND session_id = ?"
                    sql_params.append(session_id)
                
                # Filter by media_type
                if media_type:
                    sql_query += " AND media_type = ?"
                    sql_params.append(media_type)
                
                # Add text-based filtering for better matching
                query_lower = query.lower()
                search_words = query_lower.replace('what ', '').replace('do ', '').replace('i ', '').strip().split()
                for word in search_words:
                    if word:
                        sql_query += " AND (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)"
                        word_term = f"%{word}%"
                        sql_params.extend([word_term, word_term, word_term])
                
                # Add ordering and limit from settings
                max_facts = getattr(settings, 'max_facts_summary', 100)
                sql_query += f" ORDER BY timestamp DESC LIMIT {max_facts}"
                
                rows = conn.execute(sql_query, sql_params).fetchall()
                
                if not rows:
                    # Fallback: try without text filtering but keep user filtering
                    fallback_sql = """
                        SELECT id, subject, predicate, object, confidence, timestamp, frequency, context, media_type, media_data
                        FROM facts
                        WHERE 1=1
                    """
                    fallback_params = []
                    if user_profile_id:
                        fallback_sql += " AND user_profile_id = ?"
                        fallback_params.append(user_profile_id)
                    
                    if session_id:
                        fallback_sql += " AND session_id = ?"
                        fallback_params.append(session_id)
                    
                    if media_type:
                        fallback_sql += " AND media_type = ?"
                        fallback_params.append(media_type)
                    fallback_sql += f" ORDER BY timestamp DESC LIMIT {max_facts}"
                    rows = conn.execute(fallback_sql, fallback_params).fetchall()
                
                query_emb = get_embedding(query)
                scored = []
                for row in rows:
                    fact_id, subject, predicate, object_val, confidence, timestamp, frequency, context, media_type_val, media_data = row
                    fact_text = f"{subject} {predicate} {object_val}"
                    fact_emb = get_embedding(fact_text)
                    sim = self._cosine_similarity(query_emb, fact_emb)
                    scored.append((sim, fact_id, subject, predicate, object_val, confidence, timestamp, frequency, context, media_type_val, media_data))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                
                results = []
                for s in scored[:topk]:
                    if s[0] < MULTIMODAL_SIMILARITY_THRESHOLD:
                        continue
                    results.append({
                        "id": s[1],
                        "subject": s[2],
                        "predicate": s[3],
                        "object": s[4],
                        "confidence": s[5],
                        "timestamp": s[6],
                        "frequency": s[7],
                        "context": json.loads(s[8]) if s[8] else None,
                        "media_type": s[9],
                        "media_data": s[10],
                        "similarity": s[0],
                    })
                return results
        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return []

    @safe_db_operation
    def consolidate_facts(self):
        """
        Merge semantically similar facts (e.g., synonyms or high embedding similarity).
        The higher-confidence fact is kept, and frequencies are summed.
        """
        from storage.memory_utils import (_are_synonyms,
                                          calculate_agreement_score)

        with self._connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            facts = self.get_all_facts(prune_contradictions=False)
            merged = set()
            for i, fact1 in enumerate(facts):
                if fact1.id in merged:
                    continue
                for fact2 in facts[i + 1 :]:
                    if fact2.id in merged or fact1.id == fact2.id:
                        continue
                    # Only consider same subject and object
                    if (
                        fact1.subject.lower() == fact2.subject.lower()
                        and fact1.object.lower() == fact2.object.lower()
                    ):
                        # Check for synonym or high agreement in predicate
                        if (
                            _are_synonyms(fact1.predicate, fact2.predicate)
                            or calculate_agreement_score(fact1, fact2)
                            > VOLATILITY_THRESHOLDS["high"]
                        ):
                            # Merge: keep higher-confidence, sum frequency
                            if getattr(fact1, "confidence", 1.0) >= getattr(
                                fact2, "confidence", 1.0
                            ):
                                keep, drop = fact1, fact2
                            else:
                                keep, drop = fact2, fact1
                            new_freq = keep.frequency + drop.frequency
                            new_conf = min(
                                1.0,
                                max(
                                    getattr(keep, "confidence", 1.0),
                                    getattr(drop, "confidence", 1.0),
                                )
                                + DEFAULT_VALUES["confidence_decay_threshold"]
                                * 0.5
                                * new_freq,
                            )
                            cursor.execute(
                                "UPDATE facts SET frequency=?, confidence=? WHERE id=?",
                                (new_freq, new_conf, keep.id),
                            )
                            cursor.execute("DELETE FROM facts WHERE id=?", (drop.id,))
                            merged.add(drop.id)
            conn.commit()

    @safe_db_operation
    def _auto_reconcile_contradictions(
        self, new_fact: tuple, contradictions: list, emotion_score: float, conn
    ) -> Optional[int]:
        """
        Automatically reconcile contradictions using LLM merge or confidence-based suppression.

        Args:
            new_fact: Tuple of (subject, predicate, object)
            contradictions: List of (contradicting_fact, contradiction_score) tuples
            emotion_score: Emotion score of the new fact
            conn: Database connection

        Returns:
            Fact ID if reconciliation was successful, None otherwise
        """
        try:
            # Sort contradictions by score (highest first)
            contradictions.sort(key=lambda x: x[1], reverse=True)

            for conflicting_fact, contradiction_score in contradictions:
                # Skip if contradiction is too weak
                if contradiction_score < DEFAULT_VALUES["contradiction_threshold"]:
                    continue

                # Strategy 1: Confidence-based suppression
                if self._should_suppress_new_fact(
                    new_fact, conflicting_fact, emotion_score
                ):
                    # Get the conflicting fact details for user-friendly message
                    conf_fact_text = f"{conflicting_fact.subject} {conflicting_fact.predicate} {conflicting_fact.object}"
                    conf_confidence = getattr(
                        conflicting_fact, "decayed_confidence", 1.0
                    )

                    print(
                        f"💡 I didn't save that fact because your earlier statement '{conf_fact_text}' has higher confidence ({conf_confidence:.2f})."
                    )
                    return conflicting_fact.id

                # Strategy 2: LLM-assisted merging for strong contradictions
                if contradiction_score > VOLATILITY_THRESHOLDS["high"]:
                    merged_fact = self.llm_merge_beliefs(
                        {
                            "subject": new_fact[0],
                            "predicate": new_fact[1],
                            "object": new_fact[2],
                        },
                        {
                            "subject": conflicting_fact.subject,
                            "predicate": conflicting_fact.predicate,
                            "object": conflicting_fact.object,
                        },
                    )

                    if merged_fact and merged_fact.get("merged"):
                        # Update the existing fact with merged content
                        conn.execute(
                            "UPDATE facts SET subject=?, predicate=?, object=?, contradiction_score=?, volatility_score=? WHERE id=?",
                            (
                                merged_fact["subject"],
                                merged_fact["predicate"],
                                merged_fact["object"],
                                DEFAULT_VALUES["confidence_decay_threshold"],
                                conflicting_fact.volatility_score
                                * DEFAULT_VALUES["contradiction_threshold"],
                                conflicting_fact.id,
                            ),
                        )
                        print(
                            f"🤖 Auto-merged conflicting facts: {merged_fact['summary']}"
                        )
                        return conflicting_fact.id

                # Strategy 3: Keep both but mark as volatile
                print(
                    f"🤖 Keeping both facts but marking as volatile (contradiction score: {contradiction_score:.2f})"
                )
                return None

        except Exception as e:
            print(f"❌ Auto-reconciliation failed: {e}")
            return None

        return None

    @safe_db_operation
    def _should_suppress_new_fact(
        self, new_fact: tuple, existing_fact, emotion_score: float
    ) -> bool:
        """
        Determine if new fact should be suppressed based on confidence comparison.

        Args:
            new_fact: New fact tuple
            existing_fact: Existing fact object
            emotion_score: Emotion score of new fact

        Returns:
            True if new fact should be suppressed
        """
        # Calculate confidence scores
        existing_confidence = getattr(existing_fact, "decayed_confidence", 1.0)
        existing_frequency = existing_fact.frequency

        # New fact confidence based on emotion and frequency
        new_confidence = DEFAULT_VALUES["contradiction_threshold"] + (
            emotion_score * VOLATILITY_THRESHOLDS["stable"]
        )  # Base confidence + emotion boost

        # Existing fact gets bonus for frequency and lower volatility
        existing_bonus = min(
            existing_frequency * DEFAULT_VALUES["confidence_decay_threshold"],
            VOLATILITY_THRESHOLDS["stable"],
        )  # Max stable threshold bonus for frequency
        volatility_penalty = existing_fact.volatility_score * (
            DEFAULT_VALUES["confidence_decay_threshold"] * 2
        )  # Penalty for volatility

        adjusted_existing_confidence = (
            existing_confidence + existing_bonus - volatility_penalty
        )

        # Suppress if existing fact is significantly more confident
        return adjusted_existing_confidence > (
            new_confidence + DEFAULT_VALUES["confidence_difference_threshold"]
        )

    @safe_db_operation
    def get_all_facts(
        self, downrank_contradictions: bool = False, prune_contradictions: bool = True
    ) -> list:
        """
        Get all triplet facts as TripletFact objects with contradiction and volatility scores.
        Optionally downrank/prune facts with high contradiction scores.
        """
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, subject, predicate, object, source_message_id, timestamp, frequency, contradiction_score, volatility_score, confidence, user_profile_id, session_id FROM facts ORDER BY timestamp DESC"
            ).fetchall()
            facts = []
            malformed_count = 0
            for row in rows:
                # Defensive: skip malformed facts
                subject, predicate, object_ = row[1:4]
                if not all([subject, predicate, object_]):
                    malformed_count += 1
                    if malformed_count <= 3:  # Only show first few warnings
                        print(
                            f"{CONFIDENCE_ICONS['medium']} Skipping malformed fact with nulls: {row}"
                        )
                    continue
                facts.append(TripletFact(*row))

            if malformed_count > 3:
                print(
                    f"{CONFIDENCE_ICONS['medium']} ... and {malformed_count - 3} more malformed facts skipped"
                )

            # Optionally filter/prune here
            if prune_contradictions:
                facts = [f for f in facts if f.contradiction_score <= 0.5]
            return facts

    @safe_db_operation
    def format_context_for_llm(self, entries: List[MemoryEntry]) -> str:
        """
        Format memory entries for LLM context

        Args:
            entries: List of memory entries

        Returns:
            Formatted context string
        """
        context_parts = []
        for entry in entries:
            context_parts.append(f"{entry.role}: {entry.content}")

        return "\n".join(context_parts)

    @safe_db_operation
    def format_facts_for_llm(self, facts: List[TripletFact]) -> str:
        """
        Format facts for LLM context with confidence indicators and volatility

        Args:
            facts: List of facts to format

        Returns:
            Formatted facts string with confidence indicators and volatility
        """
        if not facts:
            return ""

        # Group facts by entity type
        entity_groups = {}
        for fact in facts:
            if fact.subject not in entity_groups:
                entity_groups[fact.subject] = []
            entity_groups[fact.subject].append(fact)

        # Format with confidence indicators and volatility
        formatted_parts = []
        for entity, entity_facts in entity_groups.items():
            formatted_parts.append(f"\n{entity.replace('_', ' ').title()}:")
            for fact in entity_facts:
                # Use centralized formatter with fact object
                line = format_fact_line(fact)
                formatted_parts.append(f"  {line}")

        return "\n".join(formatted_parts)

    @safe_db_operation
    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get statistics about the memory database

        Returns:
            Dictionary with memory statistics
        """
        with self._connection_pool.get_connection() as conn:
            total_messages = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            user_messages = conn.execute(
                "SELECT COUNT(*) FROM memory WHERE role = 'user'"
            ).fetchone()[0]
            assistant_messages = conn.execute(
                "SELECT COUNT(*) FROM memory WHERE role = 'assistant'"
            ).fetchone()[0]
            total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

            return {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_facts": total_facts,
            }

    @safe_db_operation
    def dump_all_memory(self) -> Dict:
        """
        Dump all memory contents for debugging

        Returns:
            Dictionary containing all memory data
        """
        with self._connection_pool.get_connection() as conn:
            # Get all memory entries
            memory_rows = conn.execute(
                """
                SELECT id, timestamp, role, content, tags
                FROM memory
                ORDER BY timestamp
            """
            ).fetchall()

            # Get all facts
            fact_rows = conn.execute(
                """
                SELECT id, subject, predicate, object, source_message_id, timestamp, frequency
                FROM facts
                ORDER BY timestamp
            """
            ).fetchall()

            return {
                "memory_entries": [
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "role": row[2],
                        "content": row[3],
                        "tags": json.loads(row[4]) if row[4] else None,
                    }
                    for row in memory_rows
                ],
                "facts": [
                    {
                        "id": row[0],
                        "subject": row[1],
                        "predicate": row[2],
                        "object": row[3],
                        "source_message_id": row[4],
                        "timestamp": row[5],
                        "frequency": row[6],
                    }
                    for row in fact_rows
                ],
            }

    @safe_db_operation
    def get_fact_history(self, fact_id: int) -> List[Dict]:
        """
        Get complete history of changes for a fact

        Args:
            fact_id: ID of the fact

        Returns:
            List of dictionaries with timestamp and fact data
        """
        with self._connection_pool.get_connection() as conn:
            # For now, return a simple history based on the fact's current state
            # In a full implementation, this would track all changes from fact_history table
            fact = conn.execute(
                """
                SELECT subject, predicate, object, timestamp, frequency
                FROM facts
                WHERE id = ?
            """,
                (fact_id,),
            ).fetchone()

            if not fact:
                return []

            # Create a simple history entry for the current fact
            return [
                {
                    "timestamp": fact[3],
                    "subject": fact[0],
                    "predicate": fact[1],
                    "object": fact[2],
                    "frequency": fact[4],
                }
            ]

    @safe_db_operation
    def restore_fact_value(self, fact_id: int, target_value: str) -> bool:
        """
        Restore a fact to a previous value from its history

        Args:
            fact_id: ID of the fact to restore
            target_value: The value to restore to

        Returns:
            True if restoration was successful, False otherwise
        """
        with self._connection_pool.get_connection() as conn:
            # Check if the target value exists in history
            history = conn.execute(
                """
                SELECT old_value, new_value, timestamp
                FROM fact_history
                WHERE fact_id = ? AND (old_value = ? OR new_value = ?)
                ORDER BY timestamp DESC
            """,
                (fact_id, target_value, target_value),
            ).fetchone()

            if not history:
                return False

            # Update the fact to the target value
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                UPDATE facts
                SET value = ?, timestamp = ?
                WHERE id = ?
            """,
                (target_value, current_time, fact_id),
            )

            # Record this restoration in history
            current_value = conn.execute(
                "SELECT value FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()[0]
            if current_value != target_value:
                conn.execute(
                    """
                    INSERT INTO fact_history (fact_id, old_value, new_value, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (fact_id, current_value, target_value, current_time),
                )

            conn.commit()
            return True

    @safe_db_operation
    def delete_fact(self, fact_id: int) -> bool:
        """
        Delete a specific fact by its ID.

        Args:
            fact_id: The ID of the fact to delete

        Returns:
            True if fact was deleted, False if not found
        """
        with self._connection_pool.get_connection() as conn:
            # Check if fact exists
            existing = conn.execute(
                "SELECT id FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()

            if not existing:
                return False

            # Delete the fact
            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))

            # Also delete any fact history for this fact
            conn.execute("DELETE FROM fact_history WHERE fact_id = ?", (fact_id,))

            conn.commit()
            return True

    @safe_db_operation
    def list_facts_with_ids(self) -> list:
        """
        Get all facts with their IDs for display and selection.
        Shows normalized subjects for debugging.

        Returns:
            List of (id, TripletFact) tuples with normalized subject info
        """
        from storage.memory_utils import _normalize_subject

        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, subject, predicate, object, source_message_id, timestamp, frequency, contradiction_score, volatility_score, confidence, last_reinforced FROM facts ORDER BY timestamp DESC"
            ).fetchall()
            facts = []
            for row in rows:
                fact = TripletFact(*row[1:9])
                # Add normalized subject for debugging
                fact.normalized_subject = _normalize_subject(fact.subject)
                facts.append((row[0], fact))
            return facts

    @safe_db_operation
    def log_contradiction(
        self,
        fact_a_id: int,
        fact_b_id: int,
        fact_a_text: str,
        fact_b_text: str,
        confidence: float,
    ) -> int:
        """
        Log a contradiction between two facts and increment volatility scores.

        Args:
            fact_a_id: ID of first fact
            fact_b_id: ID of second fact
            fact_a_text: Text representation of first fact
            fact_b_text: Text representation of second fact
            confidence: Confidence in the contradiction detection

        Returns:
            ID of the created contradiction entry
        """
        with self._connection_pool.get_connection() as conn:
            # Log the contradiction
            cursor = conn.execute(
                "INSERT INTO contradictions (fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence) VALUES (?, ?, ?, ?, ?)",
                (fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence),
            )
            contradiction_id = cursor.lastrowid

            # Increment volatility scores for both facts
            volatility_increment = DEFAULT_VALUES[
                "confidence_decay_threshold"
            ]  # Base increment per contradiction
            volatility_increment *= confidence  # Scale by confidence

            # Update fact A volatility
            conn.execute(
                "UPDATE facts SET volatility_score = volatility_score + ? WHERE id = ?",
                (volatility_increment, fact_a_id),
            )

            # Update fact B volatility
            conn.execute(
                "UPDATE facts SET volatility_score = volatility_score + ? WHERE id = ?",
                (volatility_increment, fact_b_id),
            )

            conn.commit()
            return contradiction_id

    @safe_db_operation
    def get_contradictions(self, resolved: bool = None) -> list:
        """
        Get contradiction instances from the database.

        Args:
            resolved: If True, return only resolved contradictions. If False, return only unresolved. If None, return all.

        Returns:
            List of contradiction dictionaries
        """
        with self._connection_pool.get_connection() as conn:
            if resolved is None:
                rows = conn.execute(
                    "SELECT id, fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence, timestamp, resolved, resolution_notes FROM contradictions ORDER BY timestamp DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence, timestamp, resolved, resolution_notes FROM contradictions WHERE resolved = ? ORDER BY timestamp DESC",
                    (resolved,),
                ).fetchall()

            contradictions = []
            for row in rows:
                contradictions.append(
                    {
                        "id": row[0],
                        "fact_a_id": row[1],
                        "fact_b_id": row[2],
                        "fact_a_text": row[3],
                        "fact_b_text": row[4],
                        "confidence": row[5],
                        "timestamp": row[6],
                        "resolved": bool(row[7]),
                        "resolution_notes": row[8],
                    }
                )

            return contradictions

    @safe_db_operation
    def resolve_contradiction(
        self, contradiction_id: int, resolution_notes: str
    ) -> bool:
        """
        Mark a contradiction as resolved.

        Args:
            contradiction_id: ID of the contradiction to resolve
            resolution_notes: Notes about how the contradiction was resolved

        Returns:
            True if contradiction was resolved, False if not found
        """
        with self._connection_pool.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE contradictions SET resolved = TRUE, resolution_notes = ? WHERE id = ?",
                (resolution_notes, contradiction_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    @safe_db_operation
    def calculate_temporal_decay(
        self,
        confidence: float,
        last_reinforced: str,
        decay_rate: float = DEFAULT_VALUES["confidence_decay_threshold"],
    ) -> float:
        """
        Calculate temporal decay for a fact's confidence score.

        Args:
            confidence: Base confidence score
            last_reinforced: Timestamp of last reinforcement
            decay_rate: Rate of decay per day

        Returns:
            Decayed confidence score
        """
        try:
            from datetime import datetime

            last_time = datetime.fromisoformat(last_reinforced.replace("Z", "+00:00"))
            current_time = datetime.now()
            days_since = (current_time - last_time).days

            # Apply exponential decay
            decayed_confidence = confidence * (1.0 - decay_rate) ** days_since
            return max(0.0, min(1.0, decayed_confidence))
        except Exception as e:
            print(f"Warning: Temporal decay calculation failed: {e}")
            return confidence

    @safe_db_operation
    def calculate_volatility_decay(
        self,
        confidence: float,
        volatility_score: float,
        volatility_weight: float = VOLATILITY_THRESHOLDS["stable"],
    ) -> float:
        """
        Calculate confidence decay based on volatility score.
        Facts that flip often should decay faster.

        Args:
            confidence: Base confidence score
            volatility_score: Volatility score (0.0-1.0+)
            volatility_weight: Weight for volatility impact (0.0-1.0)

        Returns:
            Decayed confidence score incorporating volatility
        """
        # Apply volatility penalty: higher volatility = faster decay
        volatility_penalty = volatility_weight * min(volatility_score, 1.0)
        decayed_confidence = confidence * (1.0 - volatility_penalty)
        return max(0.0, min(1.0, decayed_confidence))

    @safe_db_operation
    def get_facts_with_volatility_decay(
        self, volatility_weight: float = VOLATILITY_THRESHOLDS["stable"]
    ) -> list:
        """
        Get all facts with confidence scores that incorporate both temporal and volatility decay.

        Args:
            volatility_weight: Weight for volatility impact on confidence

        Returns:
            List of TripletFact objects with decayed_confidence attribute
        """
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, subject, predicate, object, source_message_id, timestamp, 
                       frequency, contradiction_score, volatility_score, confidence, last_reinforced
                FROM facts 
                ORDER BY timestamp DESC
            """
            ).fetchall()

            facts = []
            for row in rows:
                (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    source_id,
                    timestamp,
                    frequency,
                    contradiction_score,
                    volatility_score,
                    confidence,
                    last_reinforced,
                ) = row

                # Calculate temporal decay
                temporal_decay_confidence = self.calculate_temporal_decay(
                    confidence, last_reinforced
                )

                # Apply volatility decay on top of temporal decay
                final_confidence = self.calculate_volatility_decay(
                    temporal_decay_confidence, volatility_score, volatility_weight
                )

                fact = TripletFact(
                    id=fact_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_val,
                    source_message_id=source_id,
                    timestamp=timestamp,
                    frequency=frequency,
                    contradiction_score=contradiction_score,
                    volatility_score=volatility_score,
                )
                fact.decayed_confidence = final_confidence
                facts.append(fact)

            return facts

    @safe_db_operation
    def group_facts_by_subject(self, min_facts: int = 3) -> dict:
        """
        Group facts by subject and filter groups with minimum fact count.

        Args:
            min_facts: Minimum number of facts required for summarization

        Returns:
            Dictionary mapping subjects to lists of facts
        """
        all_facts = self.get_all_facts()
        subject_groups = {}

        for fact in all_facts:
            if fact.subject not in subject_groups:
                subject_groups[fact.subject] = []
            subject_groups[fact.subject].append(fact)

        # Filter groups with minimum fact count
        return {
            subject: facts
            for subject, facts in subject_groups.items()
            if len(facts) >= min_facts
        }

    @safe_db_operation
    def generate_summary_for_subject(self, subject: str, facts: list) -> str:
        """
        Generate a summary for a group of facts about a specific subject using LLM.

        Args:
            subject: The subject to summarize
            facts: List of TripletFact objects about the subject

        Returns:
            Generated summary text
        """
        try:
            import requests

            # Format facts for the prompt
            fact_texts = []
            for fact in facts:
                fact_texts.append(
                    f"- {fact.subject} {fact.predicate} {fact.object} (confidence: {getattr(fact, 'decayed_confidence', 1.0):.2f})"
                )

            facts_text = "\n".join(fact_texts)

            prompt = f"""Summarize all the following beliefs about {subject} into a coherent, natural summary:

{facts_text}

Write a concise summary that captures the key beliefs and patterns about {subject}. Focus on the most confident and frequently mentioned facts. Be natural and conversational.

Summary:"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()

            summary = response.json()["response"].strip()
            return summary

        except Exception as e:
            logging.exception(f"Warning: LLM summarization failed for {subject}")
            # Fallback: simple concatenation
            return f"About {subject}: " + "; ".join(
                [f"{f.subject} {f.predicate} {f.object}" for f in facts[:5]]
            )

    @safe_db_operation
    def store_summary(self, subject: str, summary_text: str, fact_count: int) -> int:
        """
        Store or update a summary for a subject.

        Args:
            subject: The subject being summarized
            summary_text: The generated summary
            fact_count: Number of facts used in the summary

        Returns:
            ID of the stored summary
        """
        import time

        with self._connection_pool.get_connection() as conn:
            # Check if summary already exists
            existing = conn.execute(
                "SELECT id FROM summaries WHERE subject = ?", (subject,)
            ).fetchone()

            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if existing:
                # Update existing summary
                conn.execute(
                    "UPDATE summaries SET summary_text = ?, fact_count = ?, last_updated = ? WHERE subject = ?",
                    (summary_text, fact_count, now, subject),
                )
                conn.commit()
                return existing[0]
            else:
                # Insert new summary
                cursor = conn.execute(
                    "INSERT INTO summaries (subject, summary_text, fact_count, last_updated) VALUES (?, ?, ?, ?)",
                    (subject, summary_text, fact_count, now),
                )
                conn.commit()
                return cursor.lastrowid

    @safe_db_operation
    def get_summary(self, subject: str) -> dict:
        """
        Get the stored summary for a subject.

        Args:
            subject: The subject to get summary for

        Returns:
            Summary dictionary or None if not found
        """
        with self._connection_pool.get_connection() as conn:
            row = conn.execute(
                "SELECT id, subject, summary_text, fact_count, last_updated, confidence FROM summaries WHERE subject = ?",
                (subject,),
            ).fetchone()

            if row:
                return {
                    "id": row[0],
                    "subject": row[1],
                    "summary_text": row[2],
                    "fact_count": row[3],
                    "last_updated": row[4],
                    "confidence": row[5],
                }
            return None

    @safe_db_operation
    def get_all_summaries(self) -> list:
        """
        Get all stored summaries.

        Returns:
            List of summary dictionaries
        """
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, subject, summary_text, fact_count, last_updated, confidence FROM summaries ORDER BY last_updated DESC"
            ).fetchall()

            summaries = []
            for row in rows:
                summaries.append(
                    {
                        "id": row[0],
                        "subject": row[1],
                        "summary_text": row[2],
                        "fact_count": row[3],
                        "last_updated": row[4],
                        "confidence": row[5],
                    }
                )

            return summaries

    @safe_db_operation
    def summarize_memory(
        self, min_facts: int = 3, force_regenerate: bool = False
    ) -> dict:
        """
        Generate summaries for all subjects with sufficient facts.

        Args:
            min_facts: Minimum number of facts required for summarization
            force_regenerate: If True, regenerate all summaries even if they exist

        Returns:
            Dictionary mapping subjects to their summaries
        """
        subject_groups = self.group_facts_by_subject(min_facts)
        summaries = {}

        for subject, facts in subject_groups.items():
            # Check if summary already exists and is recent
            existing_summary = self.get_summary(subject)

            if force_regenerate or not existing_summary:
                # Generate new summary
                summary_text = self.generate_summary_for_subject(subject, facts)
                summary_id = self.store_summary(subject, summary_text, len(facts))
                summaries[subject] = {
                    "text": summary_text,
                    "fact_count": len(facts),
                    "id": summary_id,
                }
            else:
                # Use existing summary
                summaries[subject] = {
                    "text": existing_summary["summary_text"],
                    "fact_count": existing_summary["fact_count"],
                    "id": existing_summary["id"],
                }

        return summaries

    @safe_db_operation
    def reinforce_fact(
        self,
        fact_id: int,
        confidence_boost: float = DEFAULT_VALUES["confidence_boost_default"],
    ) -> bool:
        """
        Reinforce a fact by updating its confidence and last_reinforced timestamp.

        Args:
            fact_id: ID of the fact to reinforce
            confidence_boost: Amount to increase confidence by

        Returns:
            True if fact was reinforced, False if not found
        """
        def do_reinforce():
            with self._connection_pool.get_connection() as conn:
                # Start transaction with IMMEDIATE lock
                conn.execute("BEGIN IMMEDIATE")
                
                try:
                    # Get current fact data
                    existing = conn.execute(
                        "SELECT confidence FROM facts WHERE id = ?", (fact_id,)
                    ).fetchone()

                    if not existing:
                        conn.rollback()
                        return False

                    current_confidence = existing[0]
                    new_confidence = min(1.0, current_confidence + confidence_boost)
                    now = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Update confidence and last_reinforced
                    conn.execute(
                        "UPDATE facts SET confidence = ?, last_reinforced = ? WHERE id = ?",
                        (new_confidence, now, fact_id),
                    )
                    conn.commit()
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    raise e

        # Execute with retry logic
        return self._connection_pool.execute_with_retry(do_reinforce)

    @safe_db_operation
    def get_facts_with_temporal_weighting(
        self, decay_rate: float = DEFAULT_VALUES["confidence_decay_threshold"]
    ) -> list:
        """
        Get all facts with temporal decay applied to their confidence scores.

        Args:
            decay_rate: Daily decay rate for confidence calculation

        Returns:
            List of TripletFact objects with decayed confidence scores
        """
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, subject, predicate, object, source_message_id, timestamp, frequency, contradiction_score, volatility_score, confidence, last_reinforced FROM facts ORDER BY timestamp DESC"
            ).fetchall()

            facts = []
            for row in rows:
                (
                    fact_id,
                    subject,
                    predicate,
                    object_,
                    source_id,
                    timestamp,
                    frequency,
                    contradiction_score,
                    volatility_score,
                    confidence,
                    last_reinforced,
                ) = row

                # Calculate decayed confidence
                decayed_confidence = self.calculate_temporal_decay(
                    confidence, last_reinforced, decay_rate
                )

                # Create TripletFact with decayed confidence
                fact = TripletFact(
                    id=fact_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    source_message_id=source_id,
                    timestamp=timestamp,
                    frequency=frequency,
                    contradiction_score=contradiction_score,
                    volatility_score=volatility_score,
                )

                # Add decayed confidence as an attribute
                fact.decayed_confidence = decayed_confidence
                fact.original_confidence = confidence
                fact.last_reinforced = last_reinforced

                facts.append(fact)

            return facts

    @safe_db_operation
    def find_similar_facts(
        self, triplet_text: str, threshold: float = CONFIDENCE_THRESHOLDS["high"]
    ) -> Optional[TripletFact]:
        """
        Find facts with high semantic similarity to the given triplet text.
        Used for fact reinforcement instead of storing duplicates.

        Args:
            triplet_text: Text representation of the fact triplet
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            Most similar TripletFact if above threshold, None otherwise
        """
        try:
            from scripts.embedder import embed

            # Get embedding for the triplet text
            triplet_embedding = embed(triplet_text)
            if np.all(triplet_embedding == 0):
                print(
                    f"Warning: Embedding failed for triplet_text '{triplet_text}' in find_similar_facts."
                )

            # Search for similar facts in memory
            with self._connection_pool.get_connection() as conn:
                # Get all facts with their embeddings
                rows = conn.execute(
                    """
                    SELECT id, subject, predicate, object, source_message_id, timestamp, 
                           frequency, contradiction_score, volatility_score, confidence, last_reinforced
                    FROM facts
                """
                ).fetchall()

                best_match = None
                best_score = 0.0

                for row in rows:
                    (
                        fact_id,
                        subject,
                        predicate,
                        object_val,
                        source_id,
                        timestamp,
                        frequency,
                        contradiction_score,
                        volatility_score,
                        confidence,
                        last_reinforced,
                    ) = row

                    # Create text representation of this fact
                    fact_text = f"{subject} {predicate} {object_val}"

                    # Calculate similarity
                    fact_embedding = embed(fact_text)
                    if np.all(fact_embedding == 0):
                        print(
                            f"Warning: Embedding failed for fact_text '{fact_text}' in find_similar_facts."
                        )
                    similarity = self._cosine_similarity(
                        triplet_embedding, fact_embedding
                    )

                    if similarity > best_score and similarity >= threshold:
                        best_score = similarity
                        best_match = TripletFact(
                            id=fact_id,
                            subject=subject,
                            predicate=predicate,
                            object=object_val,
                            source_message_id=source_id,
                            timestamp=timestamp,
                            frequency=frequency,
                            contradiction_score=contradiction_score,
                            volatility_score=volatility_score,
                        )
                        best_match.decayed_confidence = self.calculate_temporal_decay(
                            confidence, last_reinforced
                        )

                return best_match

        except Exception as e:
            print(f"Warning: Similar fact search failed: {e}")
            return None

    def _cosine_similarity(self, vec1, vec2) -> float:
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

    @safe_db_operation
    def export_memory_summary(self, output_file: str = "memory_summary.jsonl") -> bool:
        """
        Export memory summary to JSONL format with top beliefs per subject.

        Args:
            output_file: Output file path

        Returns:
            True if export successful, False otherwise
        """
        try:
            import json
            from datetime import datetime

            # Get all facts with volatility decay
            facts = self.get_facts_with_volatility_decay()

            # Group facts by subject
            subject_groups = {}
            for fact in facts:
                if fact.subject not in subject_groups:
                    subject_groups[fact.subject] = []
                subject_groups[fact.subject].append(fact)

            # Export to JSONL
            with open(output_file, "w") as f:
                for subject, subject_facts in subject_groups.items():
                    # Sort by decayed confidence (highest first)
                    sorted_facts = sorted(
                        subject_facts,
                        key=lambda f: getattr(f, "decayed_confidence", 1.0),
                        reverse=True,
                    )

                    # Take top 5 facts per subject
                    top_facts = sorted_facts[:5]

                    summary_entry = {
                        "subject": subject,
                        "fact_count": len(subject_facts),
                        "top_facts": [
                            {
                                "predicate": fact.predicate,
                                "object": fact.object,
                                "confidence": getattr(fact, "decayed_confidence", 1.0),
                                "frequency": fact.frequency,
                                "volatility": fact.volatility_score,
                                "contradiction_score": fact.contradiction_score,
                                "last_seen": fact.timestamp,
                            }
                            for fact in top_facts
                        ],
                        "export_timestamp": datetime.now().isoformat(),
                    }

                    f.write(json.dumps(summary_entry) + "\n")

            logging.info(
                f"{CONFIDENCE_ICONS['success']} Memory summary exported to {output_file}"
            )
            return True

        except Exception as e:
            logging.exception(f"Memory export failed: {e}")
            return False

    @safe_db_operation
    def summarize_contradictions(self, subject: str) -> str:
        """
        Generate an LLM summary of contradictions for a subject.
        Uses canonical subject matching to find all related facts.
        Grouping and lookups use normalized subject keys.
        """
        from storage.memory_utils import _normalize_subject

        facts = self.get_all_facts(prune_contradictions=False)

        # Normalize the target subject
        target = _normalize_subject(subject)

        # Group facts by normalized subject
        subject_to_facts = {}
        for f in facts:
            norm_subject = _normalize_subject(f.subject)
            if norm_subject not in subject_to_facts:
                subject_to_facts[norm_subject] = []
            subject_to_facts[norm_subject].append(f)

        subject_facts = [
            f
            for f in subject_to_facts.get(target, [])
            if f.contradiction_score > VOLATILITY_THRESHOLDS["stable"]
        ]

        if not subject_facts:
            return f"No contradictions found for subject '{subject}' (normalized: '{target}')."

        # Build a summary prompt
        prompt = "Summarize the following contradictory beliefs for the subject '{}' (normalized from '{}'):\n".format(
            target, subject
        )
        for f in subject_facts:
            norm_subject = _normalize_subject(f.subject)
            prompt += f"- {f.subject} [{norm_subject}] {f.predicate} {f.object} (confidence: {getattr(f, 'decayed_confidence', 1.0):.2f}, last: {f.timestamp.split()[0]})\n"
        prompt += "\nSummary:"
        # Use LLM to generate summary
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"[LLM error: {e}]\nPrompt was:\n{prompt}"

    @safe_db_operation
    def summarize_conflicts(self, subject: str) -> str:
        """
        Generate an LLM summary of contradictions for a subject.
        Uses canonical subject matching to find all related facts.
        Grouping and lookups use normalized subject keys.
        """
        from storage.memory_utils import _normalize_subject

        facts = self.get_all_facts(prune_contradictions=False)

        # Normalize the target subject
        target = _normalize_subject(subject)

        # Group facts by normalized subject
        subject_to_facts = {}
        for f in facts:
            norm_subject = _normalize_subject(f.subject)
            if norm_subject not in subject_to_facts:
                subject_to_facts[norm_subject] = []
            subject_to_facts[norm_subject].append(f)

        subject_facts = [
            f
            for f in subject_to_facts.get(target, [])
            if f.contradiction_score > VOLATILITY_THRESHOLDS["stable"]
        ]

        if not subject_facts:
            return f"No contradictions found for subject '{subject}' (normalized: '{target}')."

        # Build a summary prompt
        prompt = "Summarize the following contradictory beliefs for the subject '{}' (normalized from '{}'):\n".format(
            target, subject
        )
        for f in subject_facts:
            norm_subject = _normalize_subject(f.subject)
            prompt += f"- {f.subject} [{norm_subject}] {f.predicate} {f.object} (confidence: {getattr(f, 'decayed_confidence', 1.0):.2f}, last: {f.timestamp.split()[0]})\n"
        prompt += "\nSummary:"
        # Use LLM to generate summary
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"[LLM error: {e}]\nPrompt was:\n{prompt}"

    @safe_db_operation
    def summarize_conflicts_llm(self, subject: str) -> str:
        """
        Groups all contradictions around a subject, ranks by severity, and summarizes using Mistral.
        Returns a reconciliation question like "You've said both 'I love pizza' and 'pizza made me sick'. Should I reconcile?"
        """
        from storage.memory_utils import _normalize_subject

        facts = self.get_all_facts(prune_contradictions=False)

        # Normalize the target subject
        target = _normalize_subject(subject)

        # Group facts by normalized subject
        subject_to_facts = {}
        for f in facts:
            norm_subject = _normalize_subject(f.subject)
            if norm_subject not in subject_to_facts:
                subject_to_facts[norm_subject] = []
            subject_to_facts[norm_subject].append(f)

        # Get facts for this subject with contradictions
        subject_facts = [
            f
            for f in subject_to_facts.get(target, [])
            if f.contradiction_score > VOLATILITY_THRESHOLDS["stable"]
        ]

        if not subject_facts:
            return f"No contradictions found for subject '{subject}'."

        # Sort by contradiction score (highest first)
        subject_facts.sort(key=lambda f: f.contradiction_score, reverse=True)

        # Build the reconciliation prompt
        prompt = f"You've said contradictory things about '{subject}'. Here are the conflicting statements:\n\n"

        for i, fact in enumerate(subject_facts[:5], 1):  # Top 5 most contradictory
            confidence = getattr(fact, "decayed_confidence", 1.0)
            prompt += f'{i}. "{fact.subject} {fact.predicate} {fact.object}" (confidence: {confidence:.2f})\n'

        prompt += f"\nShould I reconcile these contradictions? If so, which statement should I keep and which should I forget?"

        # Use Mistral to generate reconciliation question
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=20,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"[LLM error: {e}]\nPrompt was:\n{prompt}"

    @safe_db_operation
    def highlight_conflicts(self) -> dict:
        """
        Highlight subjects with high contradiction scores, using normalized subject keys for grouping.
        Returns a dict of {normalized_subject: {...}}
        """
        from storage.memory_utils import _normalize_subject

        facts = self.get_all_facts(prune_contradictions=False)
        subject_to_facts = {}
        for f in facts:
            norm_subject = _normalize_subject(f.subject)
            if norm_subject not in subject_to_facts:
                subject_to_facts[norm_subject] = []
            subject_to_facts[norm_subject].append(f)

        conflicts = {}
        for norm_subject, facts in subject_to_facts.items():
            conflicting_facts = [
                fact
                for fact in facts
                if fact.contradiction_score > DEFAULT_VALUES["contradiction_threshold"]
            ]
            if len(conflicting_facts) < 2:
                continue
            max_contradiction_score = max(
                fact.contradiction_score for fact in conflicting_facts
            )
            conflicts[norm_subject] = {
                "fact_count": len(conflicting_facts),
                "max_contradiction_score": max_contradiction_score,
                "conflicting_facts": [
                    {
                        "id": fact.id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "confidence": getattr(fact, "decayed_confidence", 1.0),
                        "contradiction_score": fact.contradiction_score,
                        "volatility_score": getattr(fact, "volatility_score", 0.0),
                    }
                    for fact in conflicting_facts
                ],
            }
        return conflicts

    @safe_db_operation
    def summarize_contradictions_report(self) -> str:
        """
        Generate a comprehensive contradiction summary report in table format.
        Shows all subjects with contradictions and their conflicting beliefs.

        Returns:
            Formatted table string showing contradiction summary
        """
        from storage.memory_utils import _normalize_subject

        # Get all facts with contradictions
        facts = self.get_all_facts(prune_contradictions=False)
        high_contra_facts = [
            f
            for f in facts
            if f.contradiction_score > DEFAULT_VALUES["contradiction_threshold"]
        ]

        if not high_contra_facts:
            return "No contradictions found in memory."

        # Group by canonical subject
        canonical_subjects = {}
        for fact in high_contra_facts:
            canonical_subject = _normalize_subject(fact.subject)
            if canonical_subject not in canonical_subjects:
                canonical_subjects[canonical_subject] = []
            canonical_subjects[canonical_subject].append(fact)

        # Build the report
        report_lines = []
        report_lines.append(f"{CONFIDENCE_ICONS['high']} CONTRADICTION SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"{'Subject':<15} | {'Belief A':<25} | {'Belief B':<25} | {'Confidence Gap':<15} | {'Contradiction Score':<18}"
        )
        report_lines.append("-" * 80)

        for canonical_subject, subject_facts in canonical_subjects.items():
            if len(subject_facts) < 2:
                continue

            # Sort by contradiction score
            subject_facts.sort(key=lambda f: f.contradiction_score, reverse=True)

            # Find the most contradictory pair
            best_pair = None
            best_contradiction_score = 0

            for i in range(len(subject_facts)):
                for j in range(i + 1, len(subject_facts)):
                    f1, f2 = subject_facts[i], subject_facts[j]

                    # Calculate contradiction score between this pair
                    from storage.memory_utils import \
                        calculate_contradiction_score

                    pair_score = calculate_contradiction_score(f1, f2)

                    if pair_score > best_contradiction_score:
                        best_contradiction_score = pair_score
                        best_pair = (f1, f2)

            if best_pair:
                f1, f2 = best_pair
                belief_a = f"{f1.predicate} {f1.object}"[:24]
                belief_b = f"{f2.predicate} {f2.object}"[:24]

                # Calculate confidence gap
                conf_a = getattr(f1, "decayed_confidence", 1.0)
                conf_b = getattr(f2, "decayed_confidence", 1.0)
                confidence_gap = abs(conf_a - conf_b)

                report_lines.append(
                    f"{canonical_subject:<15} | {belief_a:<25} | {belief_b:<25} | {confidence_gap:<15.3f} | {best_contradiction_score:<18.3f}"
                )

        if len(report_lines) <= 4:  # Only header lines
            return "No significant contradictions found in memory."

        report_lines.append("-" * 80)
        report_lines.append(
            f"Total subjects with contradictions: {len(canonical_subjects)}"
        )
        report_lines.append(f"Total contradictory facts: {len(high_contra_facts)}")

        return "\n".join(report_lines)

    @safe_db_operation
    def show_emotive_facts(self, top_n: int = 10) -> list:
        """
        List top facts with highest emotion scores.
        """
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, emotion_score, confidence, timestamp FROM facts WHERE emotion_score IS NOT NULL ORDER BY emotion_score DESC, confidence DESC LIMIT ?",
                (top_n,),
            ).fetchall()
            facts = []
            for row in rows:
                facts.append(
                    {
                        "subject": row[0],
                        "predicate": row[1],
                        "object": row[2],
                        "emotion_score": row[3],
                        "confidence": row[4],
                        "timestamp": row[5],
                    }
                )
            return facts

    @safe_db_operation
    def reconcile_subject(self, subject_name: str, mode: str = "interactive") -> dict:
        """
        Reconcile contradictions for a specific subject using hybrid smart matching.
        Uses multiple strategies to find related facts.
        Args:
            subject_name: Subject name to reconcile
            mode: "interactive" or "auto"
        Returns:
            Dictionary with reconciliation results
        """
        try:
            from storage.memory_utils import (
                _extract_subject_hybrid_confidence, _smart_forget_subject)

            with self._connection_pool.get_connection() as conn:
                # Get all facts as tuples for smart matching
                all_facts = conn.execute("SELECT * FROM facts").fetchall()
                fact_tuples = [
                    (row[0], row[1], row[2], row[3]) for row in all_facts
                ]  # id, subject, predicate, object

                # Use smart matching to find related facts
                related_facts = _smart_forget_subject(subject_name, fact_tuples)

                if not related_facts:
                    return {
                        "reconciled_count": 0,
                        "subject": subject_name,
                        "message": f"No contradictions to reconcile for subject '{subject_name}'.",
                    }

                # Group facts by match type and confidence
                high_confidence_facts = [
                    f for f in related_facts if f[5] > VOLATILITY_THRESHOLDS["high"]
                ]  # confidence > high threshold
                medium_confidence_facts = [
                    f
                    for f in related_facts
                    if CONFIDENCE_THRESHOLDS["medium"]
                    <= f[5]
                    <= VOLATILITY_THRESHOLDS["high"]
                ]
                low_confidence_facts = [
                    f for f in related_facts if f[5] < CONFIDENCE_THRESHOLDS["medium"]
                ]

                reconciled_count = 0

                if mode == "interactive":
                    print(f"\n🔧 Reconciliation for '{subject_name}':")
                    print(f"Found {len(related_facts)} related facts:")

                    for i, (
                        fact_id,
                        subject,
                        predicate,
                        object_val,
                        match_type,
                        confidence,
                    ) in enumerate(related_facts, 1):
                        print(
                            f"  {i}. {subject} {predicate} {object_val} (match: {match_type}, confidence: {confidence:.2f})"
                        )

                    print("\nOptions:")
                    print("  d - Delete this fact")
                    print("  m - Merge with another fact")
                    print("  k - Keep this fact")
                    print("  s - Skip all remaining facts")

                    for (
                        fact_id,
                        subject,
                        predicate,
                        object_val,
                        match_type,
                        confidence,
                    ) in related_facts:
                        if confidence < DEFAULT_VALUES["contradiction_threshold"]:
                            print(
                                f"\n{CONFIDENCE_ICONS['medium']}  Low confidence match ({confidence:.2f}) - skipping: {subject} {predicate} {object_val}"
                            )
                            continue

                        action = input(
                            f"\nAction for '{subject} {predicate} {object_val}' (d/m/k/s): "
                        ).lower()

                        if action == "s":
                            break
                        elif action == "d":
                            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                            reconciled_count += 1
                            print(f"{CONFIDENCE_ICONS['success']} Deleted")
                        elif action == "k":
                            print(f"{CONFIDENCE_ICONS['success']} Kept")
                        elif action == "m":
                            # Find similar facts to merge with
                            similar_facts = [
                                f
                                for f in related_facts
                                if f[0] != fact_id
                                and f[5] > CONFIDENCE_THRESHOLDS["medium"]
                            ]
                            if similar_facts:
                                print("Similar facts to merge with:")
                                for j, (fid, s, p, o, mt, conf) in enumerate(
                                    similar_facts[:5], 1
                                ):
                                    print(
                                        f"  {j}. {s} {p} {o} (confidence: {conf:.2f})"
                                    )

                                merge_choice = input(
                                    "Choose fact to merge with (number) or 'c' to cancel: "
                                )
                                if merge_choice.isdigit() and 1 <= int(
                                    merge_choice
                                ) <= len(similar_facts):
                                    merge_fact = similar_facts[int(merge_choice) - 1]
                                    # Merge logic here
                                    print("🔄 Merged (placeholder)")
                                    reconciled_count += 1
                            else:
                                print("No similar facts found for merging")

                elif mode == "auto":
                    # Auto-reconcile based on confidence
                    for (
                        fact_id,
                        subject,
                        predicate,
                        object_val,
                        match_type,
                        confidence,
                    ) in related_facts:
                        if confidence > VOLATILITY_THRESHOLDS["high"]:
                            # High confidence - likely a good match, keep it
                            pass
                        elif confidence < DEFAULT_VALUES["contradiction_threshold"]:
                            # Low confidence - likely a false match, delete it
                            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                            reconciled_count += 1
                        else:
                            # Medium confidence - use contradiction detection
                            # This would need more sophisticated logic
                            pass

                conn.commit()

                return {
                    "reconciled_count": reconciled_count,
                    "subject": subject_name,
                    "total_facts": len(related_facts),
                    "high_confidence": len(high_confidence_facts),
                    "medium_confidence": len(medium_confidence_facts),
                    "low_confidence": len(low_confidence_facts),
                }

        except Exception as e:
            print(f"Error reconciling subject: {e}")
            return {"reconciled_count": 0, "error": str(e)}

    @safe_db_operation
    def llm_merge_beliefs(self, f1, f2) -> dict:
        """
        Use LLM to merge two beliefs into a new one.
        Returns a dict with 'predicate' and 'object'.
        """
        prompt = f"Merge these beliefs into a single, non-contradictory statement about {f1.subject}:\n1. {f1.subject} {f1.predicate} {f1.object}\n2. {f2.subject} {f2.predicate} {f2.object}\nMerged belief (predicate and object only):"
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=20,
            )
            response.raise_for_status()
            merged = response.json()["response"].strip()
            # Try to split into predicate and object
            parts = merged.split(None, 1)
            if len(parts) == 2:
                return {"predicate": parts[0], "object": parts[1]}
            else:
                return {"predicate": "merged", "object": merged}
        except Exception as e:
            return {"predicate": "merged", "object": f"[LLM error: {e}] {prompt}"}

    @safe_db_operation
    def get_current_episode_id(self, time_gap_threshold_minutes: int = 10) -> int:
        """
        Get the current episode ID, creating a new episode if the time gap is large.
        Args:
            time_gap_threshold_minutes: Minutes threshold for creating new episode
        Returns:
            Current episode ID
        """
        from datetime import datetime, timedelta

        with self._connection_pool.get_connection() as conn:
            # Get the latest episode
            latest_episode = conn.execute(
                "SELECT id, start_time FROM episodes ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not latest_episode:
                # Create initial episode
                cursor = conn.execute(
                    "INSERT INTO episodes (start_time, summary) VALUES (CURRENT_TIMESTAMP, 'Initial episode')"
                )
                conn.commit()
                return cursor.lastrowid
            episode_id, start_time = latest_episode
            # Check if we should create a new episode based on time gap
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                current_dt = datetime.now()
                time_diff = current_dt - start_dt
                if time_diff > timedelta(minutes=time_gap_threshold_minutes):
                    # Create new episode
                    cursor = conn.execute(
                        "INSERT INTO episodes (start_time, summary) VALUES (CURRENT_TIMESTAMP, ?)",
                        (f"Episode {episode_id + 1}",),
                    )
                    conn.commit()
                    return cursor.lastrowid
                else:
                    return episode_id
            except Exception as e:
                logging.exception(f"Time gap calculation failed: {e}")
                return episode_id

    @safe_db_operation
    def get_facts_with_personality_decay(
        self,
        personality: str = "neutral",
        volatility_weight: float = VOLATILITY_THRESHOLDS["stable"],
    ) -> list:
        """
        Get all facts with confidence scores that incorporate personality-adjusted decay.
        Args:
            personality: Personality type for decay adjustment
            volatility_weight: Weight for volatility impact on confidence
        Returns:
            List of TripletFact objects with personality-adjusted decayed_confidence attribute
        """
        if personality not in PERSONALITY_PROFILES:
            personality = DEFAULT_PERSONALITY
        profile = PERSONALITY_PROFILES[personality]

        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, subject, predicate, object, source_message_id, timestamp, 
                       frequency, contradiction_score, volatility_score, confidence, last_reinforced
                FROM facts 
                ORDER BY timestamp DESC
            """
            ).fetchall()
            facts = []
            for row in rows:
                (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    source_id,
                    timestamp,
                    frequency,
                    contradiction_score,
                    volatility_score,
                    confidence,
                    last_reinforced,
                ) = row
                # Calculate temporal decay
                temporal_decay_confidence = self.calculate_temporal_decay(
                    confidence, last_reinforced
                )
                # Apply personality-adjusted decay
                personality_decay_confidence = self.calculate_personality_decay(
                    temporal_decay_confidence, volatility_score, personality, subject
                )
                # Apply volatility decay on top of personality decay
                final_confidence = self.calculate_volatility_decay(
                    personality_decay_confidence, volatility_score, volatility_weight
                )
                fact = TripletFact(
                    id=fact_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_val,
                    source_message_id=source_id,
                    timestamp=timestamp,
                    frequency=frequency,
                    contradiction_score=contradiction_score,
                    volatility_score=volatility_score,
                )
                fact.decayed_confidence = final_confidence
                fact.personality = personality
                facts.append(fact)
            return facts

    @safe_db_operation
    def calculate_personality_decay(
        self,
        confidence: float,
        volatility_score: float,
        personality: str = "neutral",
        subject: str = None,
    ) -> float:
        """
        Calculate confidence decay adjusted for personality traits.
        Args:
            confidence: Base confidence score
            volatility_score: Volatility score
            personality: Personality type
            subject: Subject of the fact (for personality-specific adjustments)
        Returns:
            Personality-adjusted decayed confidence
        """
        if personality not in PERSONALITY_PROFILES:
            personality = DEFAULT_PERSONALITY
        profile = PERSONALITY_PROFILES[personality]
        decay_multiplier = profile["decay_multiplier"]
        # Apply personality-specific adjustments
        if personality == "loyal" and subject:
            # Slower decay for facts about "you" or "friends"
            if subject.lower() in ["you", "your", "friend", "friends", "family"]:
                decay_multiplier *= DEFAULT_VALUES[
                    "contradiction_threshold"
                ]  # Even slower decay for personal relationships
        elif personality == "skeptical":
            # Faster decay for facts with high contradiction scores (handled elsewhere)
            pass
        elif personality == "emotional":
            # Could check for emotional content in the fact (not implemented here)
            pass
        # Apply the personality-adjusted decay
        adjusted_confidence = confidence * decay_multiplier
        return max(0.0, min(1.0, adjusted_confidence))

    @safe_db_operation
    def forget_subject(self, subject_name: str, case_sensitive: bool = False) -> dict:
        """
        Delete all facts matching the subject name using hybrid smart matching.
        Uses multiple strategies: subject match, object match, predicate match, and semantic similarity.
        Args:
            subject_name: Subject name to match (will be normalized)
            case_sensitive: Whether to use case-sensitive matching
        Returns:
            Dictionary with deletion statistics
        """
        try:
            import json
            from datetime import datetime

            from storage.memory_utils import (
                _extract_subject_hybrid_confidence, _smart_forget_subject)

            deleted_facts = []

            with self._connection_pool.get_connection() as conn:
                # Get all facts as tuples for smart matching
                all_facts = conn.execute("SELECT * FROM facts").fetchall()
                fact_tuples = [
                    (row[0], row[1], row[2], row[3]) for row in all_facts
                ]  # id, subject, predicate, object

                # Use smart forgetting to find facts to delete
                facts_to_delete = _smart_forget_subject(subject_name, fact_tuples)

                # Process the results
                for (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    match_type,
                    confidence,
                ) in facts_to_delete:
                    # Get the full fact row for logging
                    fact_row = next(
                        (row for row in all_facts if row[0] == fact_id), None
                    )
                    if fact_row:
                        (
                            fact_id,
                            subject,
                            predicate,
                            object_val,
                            source_id,
                            timestamp,
                            frequency,
                            contradiction_score,
                            volatility_score,
                            confidence_score,
                            last_reinforced,
                            episode_id,
                            emotion_score,
                            context,
                        ) = fact_row

                        # Log the fact before deletion
                        deleted_fact = {
                            "id": fact_id,
                            "subject": subject,
                            "predicate": predicate,
                            "object": object_val,
                            "confidence": confidence_score,
                            "volatility_score": volatility_score,
                            "contradiction_score": contradiction_score,
                            "frequency": frequency,
                            "timestamp": timestamp,
                            "deleted_at": datetime.now().isoformat(),
                            "reason": "subject_forgotten",
                            "target_subject": subject_name,
                            "match_type": match_type,
                            "match_confidence": confidence,
                        }
                        deleted_facts.append(deleted_fact)

                # Delete the facts
                for (
                    fact_id,
                    subject,
                    predicate,
                    object_val,
                    match_type,
                    confidence,
                ) in facts_to_delete:
                    conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                conn.commit()

            # Log deleted facts to JSONL file
            if deleted_facts:
                try:
                    with open("deleted_facts.jsonl", "a") as f:
                        for fact in deleted_facts:
                            f.write(json.dumps(fact) + "\n")
                except Exception as e:
                    logging.warning(f"Could not log deleted facts: {e}")

            # Get match statistics
            match_stats = {}
            for (
                fact_id,
                subject,
                predicate,
                object_val,
                match_type,
                confidence,
            ) in facts_to_delete:
                if match_type not in match_stats:
                    match_stats[match_type] = {"count": 0, "avg_confidence": 0.0}
                match_stats[match_type]["count"] += 1
                match_stats[match_type]["avg_confidence"] += confidence

            # Calculate averages
            for match_type in match_stats:
                if match_stats[match_type]["count"] > 0:
                    match_stats[match_type]["avg_confidence"] /= match_stats[
                        match_type
                    ]["count"]

            return {
                "deleted_count": len(deleted_facts),
                "subject": subject_name,
                "canonical_subject": subject_name,  # Add canonical subject for consistency
                "match_statistics": match_stats,
                "deleted_facts": deleted_facts,
            }
        except Exception as e:
            logging.exception(f"Error forgetting subject: {e}")
            return {"deleted_count": 0, "error": str(e), "deleted_facts": []}

    @safe_db_operation
    def semantic_drift_check(
        self,
        subject: str,
        predicate: str = None,
        obj: str = None,
        fact_id: int = None,
        conn=None,
    ) -> float:
        """Check semantic drift for a new fact against the subject's cluster embedding. Returns drift value or None."""
        import numpy as np

        if conn is None:
            conn = get_conn(self.db_path)
        # Get cluster for subject
        row = conn.execute(
            "SELECT embedding, fact_ids FROM clusters WHERE subject=? ORDER BY timestamp DESC LIMIT 1",
            (subject,),
        ).fetchone()
        if not row:
            return None  # No cluster yet
        cluster_emb = np.frombuffer(row[0], dtype=np.float32)
        # Embed new fact
        fact_text = f"{subject} {predicate} {obj}"
        new_emb = embed(fact_text)
        if np.all(new_emb == 0):
            print(
                f"Warning: Embedding failed for fact_text '{fact_text}' in semantic_drift_check."
            )
        # Compute cosine drift
        dot = np.dot(cluster_emb, new_emb)
        norm1 = np.linalg.norm(cluster_emb)
        norm2 = np.linalg.norm(new_emb)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        drift = 1.0 - (dot / (norm1 * norm2))
        return drift

    @safe_db_operation
    def drift_check(self, subject: str) -> float:
        """Manually trigger drift analysis for a subject. Returns drift value or None."""
        with self._connection_pool.get_connection() as conn:
            # Get latest fact for subject
            row = conn.execute(
                "SELECT predicate, object, id FROM facts WHERE subject=? ORDER BY timestamp DESC LIMIT 1",
                (subject,),
            ).fetchone()
            if not row:
                print(f"[drift] No facts for subject '{subject}'")
                return None
            pred, obj, fact_id = row
            drift = self.semantic_drift_check(subject, pred, obj, fact_id, conn)
            if drift is not None:
                print(f"[drift] Drift for subject '{subject}': {drift:.2f}")
            else:
                print(f"[drift] No cluster embedding for subject '{subject}'")
            return drift

    @safe_db_operation
    def _update_cluster_for_fact(
        self, subject: str, predicate: str, object_: str, fact_id: int, conn
    ):
        """Update the cluster for a new fact."""
        import numpy as np

        from config.settings import semantic_drift_threshold

        # Get cluster for subject
        row = conn.execute(
            "SELECT id, embedding, fact_ids, cluster_size FROM clusters WHERE subject=? ORDER BY timestamp DESC LIMIT 1",
            (subject,),
        ).fetchone()

        if not row:
            # Create new cluster
            fact_text = f"{subject} {predicate} {object_}"
            cluster_emb = embed(fact_text).tobytes()
            if np.all(cluster_emb == 0):
                print(
                    f"Warning: Embedding failed for fact_text '{fact_text}' in _update_cluster_for_fact (new cluster)."
                )
            fact_ids_json = json.dumps([fact_id])
            conn.execute(
                "INSERT INTO clusters (subject, embedding, fact_ids, cluster_size, timestamp) VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)",
                (subject, cluster_emb, fact_ids_json),
            )
            print(f"[cluster] Created new cluster for subject '{subject}'")
        else:
            cluster_id, cluster_emb_blob, fact_ids_json, cluster_size = row
            cluster_emb = np.frombuffer(cluster_emb_blob, dtype=np.float32)
            fact_ids = json.loads(fact_ids_json)

            # Check drift before merging
            fact_text = f"{subject} {predicate} {object_}"
            new_fact_emb = embed(fact_text)
            if np.all(new_fact_emb == 0):
                print(
                    f"Warning: Embedding failed for fact_text '{fact_text}' in _update_cluster_for_fact (merge)."
                )
            # Calculate cosine similarity
            dot = np.dot(cluster_emb, new_fact_emb)
            norm1 = np.linalg.norm(cluster_emb)
            norm2 = np.linalg.norm(new_fact_emb)
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot / (norm1 * norm2)

            drift = 1.0 - similarity

            # Log drift event
            resolution_action = (
                "merged" if drift < semantic_drift_threshold else "new_cluster"
            )
            conn.execute(
                "INSERT INTO drift_events (subject, fact_id, drift_value, resolution_action) VALUES (?, ?, ?, ?)",
                (subject, fact_id, float(drift), resolution_action),
            )

            if drift < semantic_drift_threshold:
                # Merge into existing cluster
                new_centroid = (cluster_emb * cluster_size + new_fact_emb) / (
                    cluster_size + 1
                )
                fact_ids.append(fact_id)
                new_cluster_size = cluster_size + 1

                conn.execute(
                    "UPDATE clusters SET embedding=?, fact_ids=?, cluster_size=?, timestamp=CURRENT_TIMESTAMP WHERE id=?",
                    (
                        new_centroid.tobytes(),
                        json.dumps(fact_ids),
                        new_cluster_size,
                        cluster_id,
                    ),
                )
                print(
                    f"[cluster] Merged fact into cluster for '{subject}' (drift={drift:.3f}, size={new_cluster_size})"
                )
            else:
                # Create new cluster due to high drift
                new_cluster_emb = new_fact_emb.tobytes()
                new_fact_ids_json = json.dumps([fact_id])
                conn.execute(
                    "INSERT INTO clusters (subject, embedding, fact_ids, cluster_size, timestamp) VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)",
                    (subject, new_cluster_emb, new_fact_ids_json),
                )
                print(
                    f"[cluster] Created new cluster for '{subject}' due to drift ({drift:.3f})"
                )

    @safe_db_operation
    def _update_trust_score(
        self, subject: str, fact_id: int, contradiction_score: float, conn
    ):
        """Update trust score for a subject based on contradiction detection"""
        try:
            # Get current trust score
            cursor = conn.cursor()
            cursor.execute(
                "SELECT trust_score, fact_count FROM trust_scores WHERE subject = ?",
                (subject,),
            )
            result = cursor.fetchone()

            if result:
                current_trust, fact_count = result
                # Decrease trust based on contradiction score
                new_trust = max(
                    DEFAULT_VALUES["confidence_decay_threshold"],
                    current_trust
                    - (
                        contradiction_score
                        * DEFAULT_VALUES["confidence_decay_threshold"]
                    ),
                )
                cursor.execute(
                    "UPDATE trust_scores SET trust_score = ?, fact_count = fact_count + 1 WHERE subject = ?",
                    (new_trust, subject),
                )
            else:
                # Initialize trust score
                initial_trust = max(0.5, 1.0 - contradiction_score)
                cursor.execute(
                    "INSERT INTO trust_scores (subject, trust_score, fact_count) VALUES (?, ?, 1)",
                    (subject, initial_trust),
                )
        except Exception as e:
            print(f"❌ Error updating trust score: {e}")

    @safe_db_operation
    def _find_contradictions_for_fact(
        self, subject: str, predicate: str, object_: str, fact_id: int, conn
    ) -> List[Tuple]:
        """
        Find contradictions for a new fact.

        Args:
            subject: Subject of the new fact
            predicate: Predicate of the new fact
            object_: Object of the new fact
            fact_id: ID of the new fact
            conn: Database connection

        Returns:
            List of (contradicting_fact, contradiction_score) tuples
        """
        try:
            cursor = conn.cursor()

            # Get existing facts with same subject and object but different predicates
            cursor.execute(
                """
                SELECT id, subject, predicate, object, confidence, contradiction_score, volatility_score
                FROM facts 
                WHERE subject = ? AND object = ? AND id != ?
                ORDER BY timestamp DESC
            """,
                (subject, object_, fact_id),
            )

            existing_facts = cursor.fetchall()
            contradictions = []

            for row in existing_facts:
                existing_fact = TripletFact(*row)

                # Calculate contradiction score using semantic similarity
                from storage.memory_utils import calculate_agreement_score

                contradiction_score = 1.0 - calculate_agreement_score(
                    TripletFact(
                        fact_id, subject, predicate, object_, 1, "2024-01-01", 1
                    ),
                    existing_fact,
                )

                # Only consider strong contradictions
                if contradiction_score > DEFAULT_VALUES["contradiction_threshold"]:
                    contradictions.append((existing_fact, contradiction_score))

            return contradictions

        except Exception as e:
            print(f"❌ Error finding contradictions: {e}")
            return []

    @safe_db_operation
    def list_clusters(self) -> List[Dict]:
        """List all active clusters with their details."""
        with self._connection_pool.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, subject, cluster_size, timestamp, fact_ids 
                FROM clusters 
                ORDER BY cluster_size DESC, timestamp DESC
            """
            ).fetchall()

            clusters = []
            for row in rows:
                cluster_id, subject, cluster_size, timestamp, fact_ids_json = row
                fact_ids = json.loads(fact_ids_json) if fact_ids_json else []

                # Get sample facts for this cluster
                sample_facts = []
                if fact_ids:
                    placeholders = ",".join(
                        ["?" for _ in fact_ids[:3]]
                    )  # Show first 3 facts
                    fact_rows = conn.execute(
                        f"SELECT subject, predicate, object FROM facts WHERE id IN ({placeholders})",
                        fact_ids[:3],
                    ).fetchall()
                    sample_facts = [f"{s} {p} {o}" for s, p, o in fact_rows]

                clusters.append(
                    {
                        "id": cluster_id,
                        "subject": subject,
                        "cluster_size": cluster_size,
                        "timestamp": timestamp,
                        "sample_facts": sample_facts,
                    }
                )

            return clusters

    @safe_db_operation
    def get_trust_score(self, subject: str) -> Optional[Dict]:
        """Get trust score for a subject."""
        with self._connection_pool.get_connection() as conn:
            row = conn.execute(
                """
                SELECT trust_score, fact_count, contradiction_count, last_updated 
                FROM trust_scores WHERE subject=?
            """,
                (subject,),
            ).fetchone()

            if row:
                trust_score, fact_count, contradiction_count, last_updated = row
                return {
                    "subject": subject,
                    "trust_score": trust_score,
                    "fact_count": fact_count,
                    "contradiction_count": contradiction_count,
                    "last_updated": last_updated,
                }
            return None

    @safe_db_operation
    def get_drift_events(self, subject: str = None, limit: int = 50) -> List[Dict]:
        """Get drift events, optionally filtered by subject."""
        with self._connection_pool.get_connection() as conn:
            if subject:
                rows = conn.execute(
                    """
                    SELECT id, subject, fact_id, drift_value, timestamp, resolution_action
                    FROM drift_events 
                    WHERE subject=? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (subject, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, subject, fact_id, drift_value, timestamp, resolution_action
                    FROM drift_events 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()

            events = []
            for row in rows:
                (
                    event_id,
                    subject,
                    fact_id,
                    drift_value,
                    timestamp,
                    resolution_action,
                ) = row
                events.append(
                    {
                        "id": event_id,
                        "subject": subject,
                        "fact_id": fact_id,
                        "drift_value": drift_value,
                        "timestamp": timestamp,
                        "resolution_action": resolution_action,
                    }
                )

            return events

    @safe_db_operation
    def calculate_adaptive_reinforcement_weight(
        self, fact_id: int, base_score: float = 1.0
    ) -> float:
        """
        Calculate adaptive reinforcement weight based on trust, drift, and compression state.

        Args:
            fact_id: ID of the fact to reinforce
            base_score: Base reinforcement score

        Returns:
            Adaptive reinforcement weight
        """
        try:
            with self._connection_pool.get_connection() as conn:
                # Get fact details
                fact_row = conn.execute(
                    """
                    SELECT subject, confidence, contradiction_score, volatility_score 
                    FROM facts WHERE id=?
                """,
                    (fact_id,),
                ).fetchone()

                if not fact_row:
                    return base_score

                subject, confidence, contradiction_score, volatility_score = fact_row

                # Get trust score for subject
                trust_row = conn.execute(
                    """
                    SELECT trust_score FROM trust_scores WHERE subject=?
                """,
                    (subject,),
                ).fetchone()
                trust_score = trust_row[0] if trust_row else 1.0

                # Get recent drift events for subject
                drift_row = conn.execute(
                    """
                    SELECT AVG(drift_value) FROM drift_events 
                    WHERE subject=? AND timestamp > datetime('now', '-7 days')
                """,
                    (subject,),
                ).fetchone()
                avg_drift = drift_row[0] if drift_row[0] is not None else 0.0

                # Calculate drift score (inverse of drift - lower drift = higher score)
                drift_score = max(0.1, 1.0 - avg_drift)

                # Calculate adaptive weight
                # Formula: base_score * trust_score * drift_score * (1 - contradiction_score)
                adaptive_weight = (
                    base_score
                    * trust_score
                    * drift_score
                    * (
                        1
                        - contradiction_score
                        * DEFAULT_VALUES["contradiction_threshold"]
                    )
                )

                # Apply volatility penalty using config threshold
                from config.settings import VOLATILITY_THRESHOLDS

                if volatility_score > VOLATILITY_THRESHOLDS["medium"]:
                    adaptive_weight *= (
                        1 - volatility_score * VOLATILITY_THRESHOLDS["stable"]
                    )

                return max(
                    0.01, min(2.0, adaptive_weight)
                )  # Clamp between 0.01 and 2.0

        except Exception as e:
            logging.exception(f"Error calculating adaptive reinforcement: {e}")
            return base_score

    @safe_db_operation
    def reinforce_fact_adaptive(
        self,
        fact_id: int,
        base_boost: float = DEFAULT_VALUES["confidence_boost_default"],
    ) -> bool:
        """
        Reinforce a fact using adaptive reinforcement weights.

        Args:
            fact_id: ID of the fact to reinforce
            base_boost: Base confidence boost

        Returns:
            True if reinforcement was successful
        """
        try:
            adaptive_weight = self.calculate_adaptive_reinforcement_weight(
                fact_id, base_boost
            )

            with self._connection_pool.get_connection() as conn:
                # Get current confidence
                row = conn.execute(
                    "SELECT confidence FROM facts WHERE id=?", (fact_id,)
                ).fetchone()
                if not row:
                    return False

                current_confidence = row[0]
                new_confidence = min(1.0, current_confidence + adaptive_weight)

                # Update confidence and timestamp
                conn.execute(
                    """
                    UPDATE facts 
                    SET confidence=?, last_reinforced=CURRENT_TIMESTAMP 
                    WHERE id=?
                """,
                    (new_confidence, fact_id),
                )
                conn.commit()

                logging.info(
                    f"🔄 Adaptive reinforcement: fact {fact_id} boosted by {adaptive_weight:.3f} (weight: {adaptive_weight/base_boost:.2f}x)"
                )
                return True

        except Exception as e:
            logging.exception(f"Error in adaptive reinforcement: {e}")
            return False

    @safe_db_operation
    def get_reinforcement_analytics(self, subject: str = None) -> Dict:
        """
        Get analytics about reinforcement patterns for subjects.

        Args:
            subject: Optional subject to filter by

        Returns:
            Dictionary with reinforcement analytics
        """
        try:
            # Import config at the top of the function
            from config.settings import VOLATILITY_THRESHOLDS
            
            with self._connection_pool.get_connection() as conn:
                if subject:
                    # Get analytics for specific subject
                    rows = conn.execute(
                        """
                        SELECT f.id, f.subject, f.confidence, f.contradiction_score, f.volatility_score,
                               f.last_reinforced, ts.trust_score
                        FROM facts f
                        LEFT JOIN trust_scores ts ON f.subject = ts.subject
                        WHERE f.subject = ?
                        ORDER BY f.last_reinforced DESC
                    """,
                        (subject,),
                    ).fetchall()
                else:
                    # Get analytics for all subjects
                    rows = conn.execute(
                        """
                        SELECT f.id, f.subject, f.confidence, f.contradiction_score, f.volatility_score,
                               f.last_reinforced, ts.trust_score
                        FROM facts f
                        LEFT JOIN trust_scores ts ON f.subject = ts.subject
                        ORDER BY f.last_reinforced DESC
                        LIMIT 50
                    """
                    ).fetchall()

                analytics = {
                    "total_facts": len(rows),
                    "subjects": {},
                    "reinforcement_stats": {
                        "high_confidence": 0,
                        "low_confidence": 0,
                        "high_contradiction": 0,
                        "high_volatility": 0,
                    },
                }

                for row in rows:
                    (
                        fact_id,
                        subject,
                        confidence,
                        contradiction_score,
                        volatility_score,
                        last_reinforced,
                        trust_score,
                    ) = row

                    # Calculate adaptive weight for this fact
                    adaptive_weight = self.calculate_adaptive_reinforcement_weight(
                        fact_id, 1.0
                    )

                    # Update subject stats
                    if subject not in analytics["subjects"]:
                        analytics["subjects"][subject] = {
                            "fact_count": 0,
                            "avg_confidence": 0.0,
                            "avg_trust": 0.0,
                            "avg_adaptive_weight": 0.0,
                            "last_reinforced": None,
                        }

                    subject_stats = analytics["subjects"][subject]
                    subject_stats["fact_count"] += 1
                    subject_stats["avg_confidence"] += confidence
                    subject_stats["avg_trust"] += trust_score or 1.0
                    subject_stats["avg_adaptive_weight"] += adaptive_weight

                    if (
                        not subject_stats["last_reinforced"]
                        or last_reinforced > subject_stats["last_reinforced"]
                    ):
                        subject_stats["last_reinforced"] = last_reinforced

                    # Update global stats
                    if confidence > VOLATILITY_THRESHOLDS["high"]:
                        analytics["reinforcement_stats"]["high_confidence"] += 1
                    elif confidence < VOLATILITY_THRESHOLDS["stable"]:
                        analytics["reinforcement_stats"]["low_confidence"] += 1

                    if contradiction_score > CONFIDENCE_THRESHOLDS["medium"]:
                        analytics["reinforcement_stats"]["high_contradiction"] += 1

                    # Use config threshold for volatility classification
                    if volatility_score > VOLATILITY_THRESHOLDS["medium"]:
                        analytics["reinforcement_stats"]["high_volatility"] += 1

                # Calculate averages
                for subject_stats in analytics["subjects"].values():
                    fact_count = subject_stats["fact_count"]
                    subject_stats["avg_confidence"] /= fact_count
                    subject_stats["avg_trust"] /= fact_count
                    subject_stats["avg_adaptive_weight"] /= fact_count

                return analytics

        except Exception as e:
            logging.exception(f"Error getting reinforcement analytics: {e}")
            return {}

    @safe_db_operation
    def auto_reinforce_stable_memory(
        self,
        min_confidence: float = CONFIDENCE_THRESHOLDS["medium"],
        max_contradiction: float = VOLATILITY_THRESHOLDS["stable"],
    ) -> int:
        """
        Automatically reinforce stable, high-confidence facts with low contradictions.

        Args:
            min_confidence: Minimum confidence threshold
            max_contradiction: Maximum contradiction threshold

        Returns:
            Number of facts reinforced
        """
        try:
            with self._connection_pool.get_connection() as conn:
                # Get stable facts
                rows = conn.execute(
                    """
                    SELECT id, subject, confidence, contradiction_score, volatility_score
                    FROM facts 
                    WHERE confidence >= ? AND contradiction_score <= ?
                    ORDER BY confidence DESC, contradiction_score ASC
                    LIMIT 20
                """,
                    (min_confidence, max_contradiction),
                ).fetchall()

                reinforced_count = 0
                for row in rows:
                    (
                        fact_id,
                        subject,
                        confidence,
                        contradiction_score,
                        volatility_score,
                    ) = row

                    # Use adaptive reinforcement
                    if self.reinforce_fact_adaptive(
                        fact_id, 0.05
                    ):  # Small boost for stable facts
                        reinforced_count += 1

                print(f"🔄 Auto-reinforced {reinforced_count} stable facts")
                return reinforced_count

        except Exception as e:
            print(f"Error in auto-reinforcement: {e}")
            return 0

    @safe_db_operation
    def decay_unstable_memory(
        self,
        min_volatility: float = VOLATILITY_THRESHOLDS["high"],
        decay_rate: float = DEFAULT_VALUES["confidence_decay_threshold"],
    ) -> int:
        """
        Apply faster decay to unstable, volatile facts.

        Args:
            min_volatility: Minimum volatility threshold for decay
            decay_rate: Rate of confidence decay

        Returns:
            Number of facts decayed
        """
        try:
            with self._connection_pool.get_connection() as conn:
                # Get unstable facts
                rows = conn.execute(
                    """
                    SELECT id, subject, confidence, volatility_score, contradiction_score
                    FROM facts 
                    WHERE volatility_score >= ?
                    ORDER BY volatility_score DESC
                    LIMIT 20
                """,
                    (min_volatility,),
                ).fetchall()

                decayed_count = 0
                for row in rows:
                    (
                        fact_id,
                        subject,
                        confidence,
                        volatility_score,
                        contradiction_score,
                    ) = row

                    # Calculate decay based on volatility and contradiction
                    decay_multiplier = (
                        1.0
                        + (volatility_score * DEFAULT_VALUES["contradiction_threshold"])
                        + (contradiction_score * VOLATILITY_THRESHOLDS["stable"])
                    )
                    new_confidence = max(
                        0.01, confidence - (decay_rate * decay_multiplier)
                    )

                    # Update confidence
                    conn.execute(
                        "UPDATE facts SET confidence=? WHERE id=?",
                        (new_confidence, fact_id),
                    )
                    decayed_count += 1

                conn.commit()
                print(f"📉 Applied decay to {decayed_count} unstable facts")
                return decayed_count

        except Exception as e:
            print(f"Error in memory decay: {e}")
            return 0

    @safe_db_operation
    def generate_meta_goals(
        self, threshold: float = VOLATILITY_THRESHOLDS["stable"]
    ) -> list[str]:
        """
        Generate meta-goals for memory maintenance based on analytics.

        Args:
            threshold: Trust threshold below which maintenance is suggested

        Returns:
            List of suggested maintenance goals
        """
        goals = []
        analytics = self.get_reinforcement_analytics()

        # Check subjects with low trust scores
        for subject, stats in analytics.get("subjects", {}).items():
            trust = stats.get("avg_trust", 1.0)
            fact_count = stats.get("fact_count", 0)

            # Suggest compression for subjects with low trust and many facts
            if trust < threshold and fact_count > 5:
                goals.append(
                    f"Compress cluster for subject '{subject}' (trust: {trust:.2f}, facts: {fact_count})"
                )

            # Suggest reconciliation for very low trust subjects
            if trust < DEFAULT_VALUES["trust_threshold"] and fact_count > 2:
                goals.append(
                    f"Reconcile contradictions for subject '{subject}' (trust: {trust:.2f})"
                )

        # Check for high contradiction counts per subject
        facts = self.get_all_facts(prune_contradictions=False)
        subject_contradictions = {}
        for fact in facts:
            if fact.contradiction_score > 0.5:
                subject = fact.subject.lower()
                if subject not in subject_contradictions:
                    subject_contradictions[subject] = 0
                subject_contradictions[subject] += 1

        # Trigger reconciliation for subjects with >3 contradictions
        for subject, count in subject_contradictions.items():
            if count > 3:
                goals.append(
                    f"Run memory reconciliation for '{subject}' (contradictions: {count})"
                )

        # NEW: Contradiction Reconciliation via Sentiment Trajectory
        trajectory_goals = self._generate_trajectory_reconciliation_goals(facts)
        goals.extend(trajectory_goals)

        # Check overall memory health
        total_facts = analytics.get("total_facts", 0)
        high_contra_facts = analytics.get("reinforcement_stats", {}).get(
            "high_contradiction", 0
        )

        if (
            high_contra_facts
            > total_facts * DEFAULT_VALUES["confidence_decay_threshold"]
        ):  # More than 10% high contradiction
            goals.append(
                f"Run global contradiction reconciliation ({high_contra_facts} high-contradiction facts)"
            )

        # Check for memory drift
        drift_events = self.get_drift_events(limit=10)
        recent_drift = [
            e for e in drift_events if e.get("timestamp", "").startswith("2025-06-27")
        ]  # Today's events
        if len(recent_drift) > 5:
            goals.append(
                f"Investigate semantic drift ({len(recent_drift)} recent drift events)"
            )

        # Check for unstable memory
        unstable_facts = analytics.get("reinforcement_stats", {}).get(
            "high_volatility", 0
        )
        if unstable_facts > total_facts * (
            DEFAULT_VALUES["confidence_decay_threshold"] + 0.05
        ):  # More than 15% high volatility
            goals.append(
                f"Apply decay to unstable memory ({unstable_facts} high-volatility facts)"
            )

        # Add specific action goals for high contradiction subjects
        if subject_contradictions:
            max_contra_subject = max(subject_contradictions.items(), key=lambda x: x[1])
            if max_contra_subject[1] > 3:
                goals.append(f"compress_cluster for subject '{max_contra_subject[0]}'")
                goals.append(
                    f"run reconciliation for subject '{max_contra_subject[0]}'"
                )

        return goals

    @safe_db_operation
    def _generate_trajectory_reconciliation_goals(
        self, facts: List[TripletFact]
    ) -> List[str]:
        """
        Generate meta-goals for contradiction reconciliation based on sentiment trajectory.

        Args:
            facts: List of all facts to analyze

        Returns:
            List of trajectory-based reconciliation goals
        """
        from storage.memory_utils import detect_contradictions

        goals = []

        # Group facts by subject-object pairs
        subject_object_groups = {}
        for fact in facts:
            key = (fact.subject.lower().strip(), fact.object.lower().strip())
            if key not in subject_object_groups:
                subject_object_groups[key] = []
            subject_object_groups[key].append(fact)

        # Analyze each subject-object pair for trajectory-based reconciliation
        for (subject, object_), group_facts in subject_object_groups.items():
            if len(group_facts) < 3:  # Need enough facts for trajectory analysis
                continue

            # Get sentiment trajectory
            trajectory = self.get_sentiment_trajectory(subject, object_)
            slope = trajectory.get("slope", 0.0)

            # Check if trajectory is strong enough for reconciliation
            if abs(slope) > DEFAULT_VALUES["slope_threshold"]:
                # Find contradictions in this group
                contradictions = detect_contradictions(group_facts)

                if contradictions:
                    # Identify facts that contradict the trajectory
                    opposing_facts = self._identify_opposing_facts(group_facts, slope)

                    if opposing_facts:
                        # Create reconciliation goal
                        action = "positive" if slope > 0 else "negative"
                        goal = {
                            "action": "resolve_contradiction_by_trajectory",
                            "subject": subject,
                            "object": object_,
                            "trajectory_slope": slope,
                            "trajectory_direction": action,
                            "suggested_deletions": [f.id for f in opposing_facts],
                            "reason": f"Trajectory shows strong {action} trend (slope: {slope:.3f})",
                        }

                        goals.append(
                            f"Reconcile contradictions for '{subject} {object_}' via trajectory (slope: {slope:.3f}, {len(opposing_facts)} opposing facts)"
                        )

        return goals

    @safe_db_operation
    def _identify_opposing_facts(
        self, facts: List[TripletFact], trajectory_slope: float
    ) -> List[TripletFact]:
        """
        Identify facts that oppose the sentiment trajectory.

        Args:
            facts: List of facts for a subject-object pair
            trajectory_slope: Slope of sentiment trajectory

        Returns:
            List of facts that should be considered for deletion
        """
        from storage.memory_utils import (compute_decay_weighted_confidence,
                                          get_sentiment_score)

        opposing_facts = []

        # Determine expected sentiment direction
        expected_positive = trajectory_slope > 0

        for fact in facts:
            # Calculate sentiment score for this fact
            sentiment_score = get_sentiment_score(fact.predicate)

            # Check if fact opposes trajectory
            fact_is_positive = sentiment_score > 0
            opposes_trajectory = (expected_positive and not fact_is_positive) or (
                not expected_positive and fact_is_positive
            )

            if opposes_trajectory:
                # Calculate recency-weighted confidence
                decayed_confidence = compute_decay_weighted_confidence(
                    getattr(fact, "confidence", 1.0), fact.timestamp
                )

                # Only consider facts that are old or have low confidence
                days_old = (
                    datetime.now()
                    - datetime.strptime(fact.timestamp.split()[0], "%Y-%m-%d")
                ).days

                if (
                    days_old > DEFAULT_VALUES["days_threshold"]
                    or decayed_confidence < DEFAULT_VALUES["contradiction_threshold"]
                ):
                    opposing_facts.append(fact)

        # Sort by recency (oldest first) and confidence (lowest first)
        opposing_facts.sort(
            key=lambda f: (
                datetime.strptime(f.timestamp.split()[0], "%Y-%m-%d"),
                getattr(f, "confidence", 1.0),
            )
        )

        return opposing_facts

    @safe_db_operation
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and common artifacts"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove common artifacts
        text = re.sub(r'^["\']+|["\']+$', "", text)  # Remove quotes at start/end
        text = re.sub(
            r"^[^\w\s]+|[^\w\s]+$", "", text
        )  # Remove punctuation at start/end

        return text.strip()

    def _demote_older_facts_by_subject(self, subject: str, conn, demotion_factor: float = None) -> None:
        """
        Demote older facts about the same subject by reducing their confidence.
        
        This method automatically resolves contradictions by prioritizing the most recent fact
        for a given subject. When a new fact is inserted, all older facts with the same subject
        have their confidence reduced by the demotion_factor.
        
        Args:
            subject: The subject of the new fact being inserted
            conn: Database connection (must be within transaction)
            demotion_factor: Factor to multiply confidence by (default 0.5 = halve confidence)
        """
        # Use configured default if not specified
        if demotion_factor is None:
            demotion_factor = DEFAULT_VALUES["fact_demotion_factor"]
            
        try:
            # Get all existing facts about this subject
            cursor = conn.execute(
                """
                SELECT id, confidence, timestamp 
                FROM facts 
                WHERE subject = ? 
                ORDER BY timestamp DESC
                """,
                (subject,)
            )
            existing_facts = cursor.fetchall()
            
            if len(existing_facts) == 0:
                return  # No existing facts to demote
                
            # Demote all existing facts (except the most recent one if it's very recent)
            demoted_count = 0
            for fact_id, old_confidence, fact_timestamp in existing_facts:
                # Skip if this fact was created very recently (within last minute)
                # to avoid demoting facts that are part of the same conversation
                try:
                    fact_time = datetime.strptime(fact_timestamp, "%Y-%m-%d %H:%M:%S")
                    current_time = datetime.now()
                    time_diff = (current_time - fact_time).total_seconds()
                    
                    if time_diff < 60:  # Skip facts created within last minute
                        continue
                except (ValueError, TypeError):
                    # If timestamp parsing fails, proceed with demotion
                    pass
                
                # Calculate new confidence
                new_confidence = old_confidence * demotion_factor
                
                # Update the fact's confidence
                conn.execute(
                    """
                    UPDATE facts 
                    SET confidence = ?, 
                        last_reinforced = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (new_confidence, fact_id)
                )
                demoted_count += 1
            
            if demoted_count > 0:
                logging.info(f"Demoted {demoted_count} older facts for subject '{subject}' (confidence reduced by factor {demotion_factor})")
                
        except Exception as e:
            logging.error(f"Error demoting older facts for subject '{subject}': {e}")
            # Don't raise the exception - this is a non-critical operation
            # that shouldn't prevent the main fact insertion from succeeding

    @safe_db_operation
    def resolve_contradiction_interactive(
        self, fact1_id: int, fact2_id: int, contradiction_score: float
    ) -> bool:
        """
        Interactively resolve a contradiction by asking the user which fact is correct.

        Args:
            fact1_id: ID of first conflicting fact
            fact2_id: ID of second conflicting fact
            contradiction_score: Calculated contradiction score

        Returns:
            True if resolution was successful, False otherwise
        """
        try:
            with self._connection_pool.get_connection() as conn:
                # Get both facts
                fact1_row = conn.execute(
                    "SELECT subject, predicate, object, contradiction_score FROM facts WHERE id = ?",
                    (fact1_id,),
                ).fetchone()
                fact2_row = conn.execute(
                    "SELECT subject, predicate, object, contradiction_score FROM facts WHERE id = ?",
                    (fact2_id,),
                ).fetchone()

                if not fact1_row or not fact2_row:
                    print("❌ One or both facts not found.")
                    return False

                fact1_text = f"{fact1_row[0]} {fact1_row[1]} {fact1_row[2]}"
                fact2_text = f"{fact2_row[0]} {fact2_row[1]} {fact2_row[2]}"

                print(
                    f"\n{CONFIDENCE_ICONS['medium']} CONTRADICTION DETECTED (score: {contradiction_score:.2f}):"
                )
                print(f"A: '{fact1_text}'")
                print(f"B: '{fact2_text}'")
                print("\nWhich statement is correct?")
                print("1. A is correct (delete B)")
                print("2. B is correct (delete A)")
                print("3. Both are wrong (delete both)")
                print("4. Keep both (mark as volatile)")
                print("5. Cancel (no changes)")

                try:
                    choice = input("\nEnter your choice (1-5): ").strip()

                    if choice == "1":
                        # Delete fact B, keep fact A
                        conn.execute("DELETE FROM facts WHERE id = ?", (fact2_id,))
                        print(
                            f"{CONFIDENCE_ICONS['success']} Deleted fact B: '{fact2_text}'"
                        )
                        return True
                    elif choice == "2":
                        # Delete fact A, keep fact B
                        conn.execute("DELETE FROM facts WHERE id = ?", (fact1_id,))
                        print(
                            f"{CONFIDENCE_ICONS['success']} Deleted fact A: '{fact1_text}'"
                        )
                        return True
                    elif choice == "3":
                        # Delete both facts
                        conn.execute(
                            "DELETE FROM facts WHERE id IN (?, ?)", (fact1_id, fact2_id)
                        )
                        print(
                            f"{CONFIDENCE_ICONS['success']} Deleted both conflicting facts"
                        )
                        return True
                    elif choice == "4":
                        # Mark both as volatile
                        conn.execute(
                            "UPDATE facts SET volatility_score = ? WHERE id IN (?, ?)",
                            (VOLATILITY_THRESHOLDS["high"], fact1_id, fact2_id),
                        )
                        print(
                            f"{CONFIDENCE_ICONS['success']} Marked both facts as volatile"
                        )
                        return True
                    elif choice == "5":
                        print("❌ No changes made")
                        return False
                    else:
                        print("❌ Invalid choice")
                        return False

                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Cancelled by user")
                    return False

        except Exception as e:
            print(f"❌ Error resolving contradiction: {e}")
            return False

    @safe_db_operation
    def analyze_personality_shift(self) -> str:
        """
        Analyze memory state and determine optimal personality.

        Returns:
            Personality type: 'skeptical', 'neutral', or 'loyal'
        """
        try:
            # Get all facts for analysis
            facts = self.get_all_facts(prune_contradictions=False)

            if not facts:
                return "neutral"

            # Count contradictions per subject
            subject_contradictions = {}
            total_contradiction_score = 0
            emotional_words = 0

            for fact in facts:
                subject = fact.subject.lower()
                if subject not in subject_contradictions:
                    subject_contradictions[subject] = 0

                if fact.contradiction_score > DEFAULT_VALUES["contradiction_threshold"]:
                    subject_contradictions[subject] += 1
                    total_contradiction_score += fact.contradiction_score

                # Count emotional words
                emotional_predicates = [
                    "hate",
                    "love",
                    "despise",
                    "adore",
                    "loathe",
                    "detest",
                    "abhor",
                ]
                if fact.predicate.lower() in emotional_predicates:
                    emotional_words += 1

            # Calculate metrics
            avg_contradiction_score = (
                total_contradiction_score / len(facts) if facts else 0
            )
            max_contradictions_per_subject = (
                max(subject_contradictions.values()) if subject_contradictions else 0
            )
            emotional_ratio = emotional_words / len(facts) if facts else 0

            # Decision logic
            if (
                max_contradictions_per_subject > 3
                or avg_contradiction_score > CONFIDENCE_THRESHOLDS["medium"]
                or emotional_ratio > VOLATILITY_THRESHOLDS["stable"]
            ):
                personality = "skeptical"
            elif (
                max_contradictions_per_subject <= 1
                and avg_contradiction_score < VOLATILITY_THRESHOLDS["stable"]
                and emotional_ratio < 0.1
            ):
                personality = "loyal"
            else:
                personality = "neutral"

            # Log the decision
            self._log_personality_shift(
                personality,
                {
                    "max_contradictions_per_subject": max_contradictions_per_subject,
                    "avg_contradiction_score": avg_contradiction_score,
                    "emotional_ratio": emotional_ratio,
                    "total_facts": len(facts),
                },
            )

            return personality

        except Exception as e:
            print(f"Warning: Personality analysis failed: {e}")
            return "neutral"

    @safe_db_operation
    def _log_personality_shift(self, personality: str, metrics: dict):
        """Log personality shift decision to file"""
        try:
            import json
            import os
            from datetime import datetime

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "personality": personality,
                "metrics": metrics,
                "reason": self._get_personality_reason(personality, metrics),
            }

            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)

            # Append to personality state log
            with open("logs/personality_state.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            print(f"Warning: Failed to log personality shift: {e}")

    @safe_db_operation
    def _get_personality_reason(self, personality: str, metrics: dict) -> str:
        """Get human-readable reason for personality choice"""
        if personality == "skeptical":
            return f"High contradiction level ({metrics.get('contradiction_level', 0):.2f})"
        elif personality == "emotional":
            return f"High emotional content ({metrics.get('emotion_score', 0):.2f})"
        elif personality == "analytical":
            return "Stable memory state with low contradictions"
        else:
            return "Balanced conditions"

    @safe_db_operation
    def get_sentiment_trajectory(
        self, subject: str, object_: str = None
    ) -> Dict[str, float]:
        """
        Get sentiment trajectory (emotional arc) for a subject-object pair.

        Args:
            subject: Subject to analyze
            object_: Object to analyze (if None, analyze all objects for subject)

        Returns:
            Dictionary with trajectory metrics
        """
        from storage.memory_utils import get_sentiment_trajectory

        # Get all facts for this subject
        all_facts = self.get_all_facts(prune_contradictions=False)

        if object_:
            # Filter for specific subject-object pair
            target_facts = [
                f
                for f in all_facts
                if f.subject.lower().strip() == subject.lower().strip()
                and f.object.lower().strip() == object_.lower().strip()
            ]
        else:
            # Filter for subject only
            target_facts = [
                f
                for f in all_facts
                if f.subject.lower().strip() == subject.lower().strip()
            ]

        if len(target_facts) < 2:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "recent_sentiment": 0.0,
                "volatility": 0.0,
                "fact_count": len(target_facts),
            }

        # Compute trajectory
        trajectory = get_sentiment_trajectory(target_facts)
        trajectory["fact_count"] = len(target_facts)

        return trajectory

    @safe_db_operation
    def get_opinion_summary(self, subject: str, object_: str = None) -> Dict[str, Any]:
        """
        Get comprehensive opinion summary with tone breakdown over time.

        Args:
            subject: Subject to analyze
            object_: Object to analyze (if None, analyze all objects for subject)

        Returns:
            Dictionary with opinion analysis
        """
        from storage.memory_utils import (get_sentiment_score,
                                          get_volatility_score)

        # Get facts
        all_facts = self.get_all_facts(prune_contradictions=False)

        if object_:
            target_facts = [
                f
                for f in all_facts
                if f.subject.lower().strip() == subject.lower().strip()
                and f.object.lower().strip() == object_.lower().strip()
            ]
        else:
            target_facts = [
                f
                for f in all_facts
                if f.subject.lower().strip() == subject.lower().strip()
            ]

        if not target_facts:
            return {
                "subject": subject,
                "object": object_,
                "fact_count": 0,
                "average_sentiment": 0.0,
                "volatility": 0.0,
                "trajectory": None,
                "tone_breakdown": {},
                "recent_trend": "neutral",
            }

        # Sort by timestamp
        sorted_facts = sorted(target_facts, key=lambda f: f.timestamp)

        # Calculate sentiment scores
        sentiment_scores = [get_sentiment_score(f.predicate) for f in sorted_facts]
        average_sentiment = np.mean(sentiment_scores)

        # Calculate volatility
        volatility = get_volatility_score(sorted_facts)

        # Get trajectory
        trajectory = self.get_sentiment_trajectory(subject, object_)

        # Tone breakdown over time
        tone_breakdown = {
            "positive": len(
                [s for s in sentiment_scores if s > VOLATILITY_THRESHOLDS["stable"]]
            ),
            "negative": len(
                [s for s in sentiment_scores if s < -VOLATILITY_THRESHOLDS["stable"]]
            ),
            "neutral": len(
                [
                    s
                    for s in sentiment_scores
                    if -VOLATILITY_THRESHOLDS["stable"]
                    <= s
                    <= VOLATILITY_THRESHOLDS["stable"]
                ]
            ),
        }

        # Recent trend
        if len(sentiment_scores) >= 3:
            recent_avg = np.mean(sentiment_scores[-3:])
            if recent_avg > VOLATILITY_THRESHOLDS["stable"]:
                recent_trend = "improving"
            elif recent_avg < -VOLATILITY_THRESHOLDS["stable"]:
                recent_trend = "declining"
            else:
                recent_trend = "stable"
        else:
            recent_trend = "insufficient_data"

        return {
            "subject": subject,
            "object": object_,
            "fact_count": len(target_facts),
            "average_sentiment": average_sentiment,
            "volatility": volatility,
            "trajectory": trajectory,
            "tone_breakdown": tone_breakdown,
            "recent_trend": recent_trend,
            "sentiment_history": sentiment_scores,
        }

    @safe_db_operation
    def get_volatility_report(self, min_facts: int = 3) -> List[Dict[str, Any]]:
        """
        Get report of subjects with unstable preferences.

        Args:
            min_facts: Minimum number of facts required for analysis

        Returns:
            List of subjects with volatility scores, sorted by volatility
        """
        from storage.memory_utils import get_volatility_score

        all_facts = self.get_all_facts(prune_contradictions=False)

        # Group facts by subject
        subject_groups = {}
        for fact in all_facts:
            subject = fact.subject.lower().strip()
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(fact)

        # Calculate volatility for each subject
        volatility_data = []
        for subject, facts in subject_groups.items():
            if len(facts) >= min_facts:
                volatility = get_volatility_score(facts)
                if (
                    volatility > DEFAULT_VALUES["volatility_threshold"]
                ):  # Only include subjects with some volatility
                    volatility_data.append(
                        {
                            "subject": subject,
                            "fact_count": len(facts),
                            "volatility": volatility,
                            "facts": facts,
                        }
                    )

        # Sort by volatility (highest first)
        volatility_data.sort(key=lambda x: x["volatility"], reverse=True)

        return volatility_data

    @safe_db_operation
    def apply_confidence_decay(
        self,
        min_age_days: int = 7,
        decay_rate: float = DEFAULT_VALUES["decay_rate_default"],
    ) -> Dict[str, Any]:
        """
        Apply confidence decay to facts that haven't been reinforced recently.

        Args:
            min_age_days: Minimum age in days before applying decay
            decay_rate: Daily decay rate (default = 2% decay per day)

        Returns:
            Dictionary with decay results
        """
        logging.info(
            f"🔄 Applying confidence decay to facts older than {min_age_days} days"
        )

        results = {
            "total_facts": 0,
            "decayed_facts": 0,
            "deleted_facts": 0,
            "errors": [],
        }

        try:
            with self._connection_pool.get_connection() as conn:
                # Get all facts with their current confidence
                cursor = conn.execute(
                    """
                    SELECT id, subject, predicate, object, timestamp, confidence 
                    FROM facts 
                    WHERE confidence > DEFAULT_VALUES["confidence_decay_threshold"]
                """
                )

                facts_to_process = []
                for row in cursor.fetchall():
                    fact_id, subject, predicate, object_, timestamp, confidence = row

                    try:
                        # Calculate age in days
                        fact_date = datetime.strptime(timestamp.split()[0], "%Y-%m-%d")
                        days_old = (datetime.now() - fact_date).days

                        if days_old >= min_age_days:
                            # Calculate decay
                            decay_factor = decay_rate**days_old
                            new_confidence = confidence * decay_factor

                            facts_to_process.append(
                                {
                                    "id": fact_id,
                                    "old_confidence": confidence,
                                    "new_confidence": new_confidence,
                                    "days_old": days_old,
                                    "decay_factor": decay_factor,
                                }
                            )

                    except Exception as e:
                        logging.warning(f"Error processing fact {fact_id}: {e}")
                        results["errors"].append(
                            f"Error processing fact {fact_id}: {str(e)}"
                        )

                results["total_facts"] = len(facts_to_process)

                # Apply decay and potentially delete very low confidence facts
                for fact_data in facts_to_process:
                    try:
                        if (
                            fact_data["new_confidence"]
                            < DEFAULT_VALUES["confidence_decay_threshold"]
                        ):
                            # Delete fact with very low confidence
                            conn.execute(
                                "DELETE FROM facts WHERE id = ?", (fact_data["id"],)
                            )
                            results["deleted_facts"] += 1
                            logging.info(
                                f"🗑️ Deleted fact {fact_data['id']} (confidence: {fact_data['new_confidence']:.3f})"
                            )
                        else:
                            # Update confidence
                            conn.execute(
                                "UPDATE facts SET confidence = ? WHERE id = ?",
                                (fact_data["new_confidence"], fact_data["id"]),
                            )
                            results["decayed_facts"] += 1

                            # Log significant decays
                            if (
                                fact_data["old_confidence"]
                                - fact_data["new_confidence"]
                                > DEFAULT_VALUES["confidence_difference_threshold"]
                            ):
                                logging.info(
                                    f"📉 Decayed fact {fact_data['id']}: {fact_data['old_confidence']:.3f} → {fact_data['new_confidence']:.3f}"
                                )

                    except Exception as e:
                        logging.error(f"Error updating fact {fact_data['id']}: {e}")
                        results["errors"].append(
                            f"Error updating fact {fact_data['id']}: {str(e)}"
                        )

                # Log decay events to trace file
                self._log_decay_events(facts_to_process)

                conn.commit()

        except Exception as e:
            logging.error(f"❌ Error applying confidence decay: {e}")
            results["errors"].append(f"Decay error: {str(e)}")

        logging.info(
            f"{CONFIDENCE_ICONS['success']} Confidence decay completed: {results['decayed_facts']} decayed, {results['deleted_facts']} deleted"
        )
        return results

    @safe_db_operation
    def _log_decay_events(self, decayed_facts: List[Dict]) -> None:
        """
        Log decay events to trace file for inspection.

        Args:
            decayed_facts: List of facts that were decayed
        """
        try:
            import json
            from datetime import datetime

            trace_file = "logs/trace.jsonl"
            os.makedirs("logs", exist_ok=True)

            with open(trace_file, "a") as f:
                for fact_data in decayed_facts:
                    trace_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "confidence_decay",
                        "fact_id": fact_data["id"],
                        "old_confidence": fact_data["old_confidence"],
                        "new_confidence": fact_data["new_confidence"],
                        "days_old": fact_data["days_old"],
                        "decay_factor": fact_data["decay_factor"],
                    }
                    f.write(json.dumps(trace_entry) + "\n")

        except Exception as e:
            logging.warning(f"Could not log decay events: {e}")

    @safe_db_operation
    def prune_memory(
        self,
        confidence_threshold: float = VOLATILITY_THRESHOLDS["stable"],
        age_threshold_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Prune memory by removing low-confidence and old facts.

        Args:
            confidence_threshold: Minimum confidence to keep fact
            age_threshold_days: Minimum age in days before considering deletion

        Returns:
            Dictionary with pruning results
        """
        logging.info(
            f"🧹 Pruning memory (confidence < {confidence_threshold}, age > {age_threshold_days} days)"
        )

        results = {"total_facts": 0, "pruned_facts": 0, "errors": []}

        try:
            with self._connection_pool.get_connection() as conn:
                # Get facts to consider for pruning
                cursor = conn.execute(
                    """
                    SELECT id, subject, predicate, object, timestamp, confidence 
                    FROM facts 
                    WHERE confidence < ? OR confidence IS NULL
                """,
                    (confidence_threshold,),
                )

                facts_to_evaluate = []
                for row in cursor.fetchall():
                    fact_id, subject, predicate, object_, timestamp, confidence = row

                    try:
                        # Calculate age
                        fact_date = datetime.strptime(timestamp.split()[0], "%Y-%m-%d")
                        days_old = (datetime.now() - fact_date).days

                        if days_old >= age_threshold_days:
                            facts_to_evaluate.append(
                                {
                                    "id": fact_id,
                                    "subject": subject,
                                    "predicate": predicate,
                                    "object": object_,
                                    "confidence": confidence or 0.0,
                                    "days_old": days_old,
                                }
                            )

                    except Exception as e:
                        logging.warning(f"Error evaluating fact {fact_id}: {e}")
                        results["errors"].append(
                            f"Error evaluating fact {fact_id}: {str(e)}"
                        )

                results["total_facts"] = len(facts_to_evaluate)

                # Prune facts
                for fact_data in facts_to_evaluate:
                    try:
                        conn.execute(
                            "DELETE FROM facts WHERE id = ?", (fact_data["id"],)
                        )
                        results["pruned_facts"] += 1
                        logging.info(
                            f"🗑️ Pruned fact {fact_data['id']}: '{fact_data['subject']} {fact_data['predicate']} {fact_data['object']}' (confidence: {fact_data['confidence']:.3f}, age: {fact_data['days_old']} days)"
                        )

                    except Exception as e:
                        logging.error(f"Error pruning fact {fact_data['id']}: {e}")
                        results["errors"].append(
                            f"Error pruning fact {fact_data['id']}: {str(e)}"
                        )

                conn.commit()

        except Exception as e:
            logging.error(f"❌ Error pruning memory: {e}")
            results["errors"].append(f"Pruning error: {str(e)}")

        logging.info(
            f"{CONFIDENCE_ICONS['success']} Memory pruning completed: {results['pruned_facts']} facts removed"
        )
        return results

    @safe_db_operation
    def update_fact_causal_info(self, fact_id: str, cause: str, causal_strength: float):
        """Update a fact with causal linkage information."""
        try:
            with self._connection_pool.get_connection() as conn:
                # Check if we have enhanced_facts table (Phase 2)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='enhanced_facts'"
                )
                has_enhanced_facts = cursor.fetchone() is not None
                
                if has_enhanced_facts:
                    # Update in enhanced_facts table
                    conn.execute(
                        """UPDATE enhanced_facts 
                           SET change_history = CASE 
                               WHEN change_history IS NULL OR change_history = '' 
                               THEN json_array(json_object('action', 'causal_link_added', 'cause', ?, 'strength', ?, 'timestamp', ?))
                               ELSE json_insert(change_history, '$[#]', json_object('action', 'causal_link_added', 'cause', ?, 'strength', ?, 'timestamp', ?))
                           END
                           WHERE id = ?""",
                        (cause, causal_strength, time.time(), cause, causal_strength, time.time(), fact_id)
                    )
                else:
                    # For basic facts table, add a note in context field
                    import json
                    causal_info = {
                        'cause': cause,
                        'causal_strength': causal_strength,
                        'timestamp': time.time()
                    }
                    conn.execute(
                        """UPDATE facts 
                           SET context = CASE 
                               WHEN context IS NULL 
                               THEN ?
                               ELSE context || ' | Causal: ' || ?
                           END
                           WHERE id = ?""",
                        (json.dumps(causal_info), json.dumps(causal_info), fact_id)
                    )
                
                conn.commit()
                print(f"[MemoryLog] Updated fact {fact_id} with causal info: {cause} (strength: {causal_strength:.3f})")
                
        except Exception as e:
            print(f"[MemoryLog] Error updating causal info for fact {fact_id}: {e}")

    @safe_db_operation
    def analyze_emotional_stability(self, subject: str) -> str:
        """
        Analyze emotional stability based on sentiment trajectory and volatility using dynamic clustering.
        Replaces hardcoded slope thresholds with data-driven personality inference.

        Args:
            subject: Subject to analyze (e.g., "i", "user")

        Returns:
            'stable', 'fluctuating', 'loyal', 'skeptical', or 'emotional' based on trajectory analysis
        """
        from config.settings import (PERSONALITY_ADAPTATION,
                                     PERSONALITY_CLUSTERING)

        try:
            trajectory = self.get_sentiment_trajectory(subject)

            # Check if we have enough data for analysis
            min_facts = PERSONALITY_ADAPTATION["min_facts_for_analysis"]
            if trajectory["fact_count"] < min_facts:
                return "stable"  # Default to stable if insufficient data

            volatility = trajectory["volatility"]
            slope = trajectory["slope"]
            r_squared = trajectory["r_squared"]
            recent_sentiment = trajectory["recent_sentiment"]

            # Dynamic clustering based on multiple metrics
            # Use K-means-like approach with predefined personality clusters

            # Calculate personality scores
            stability_score = 1.0 - volatility  # Higher = more stable
            trend_strength = (
                abs(slope) * r_squared
            )  # How strong and consistent the trend is
            sentiment_intensity = abs(
                recent_sentiment
            )  # How strong current feelings are

            # Get dynamic thresholds from settings
            stability_threshold = PERSONALITY_CLUSTERING["stability_threshold"]
            volatility_threshold = PERSONALITY_CLUSTERING["volatility_threshold"]
            sentiment_intensity_threshold = PERSONALITY_CLUSTERING[
                "sentiment_intensity_threshold"
            ]
            loyalty_positive_slope = PERSONALITY_CLUSTERING["loyalty_positive_slope"]
            loyalty_r_squared = PERSONALITY_CLUSTERING["loyalty_r_squared"]
            skeptical_negative_slope = PERSONALITY_CLUSTERING[
                "skeptical_negative_slope"
            ]
            skeptical_volatility = PERSONALITY_CLUSTERING["skeptical_volatility"]

            # Personality classification using dynamic thresholds
            if (
                stability_score > stability_threshold
                and trend_strength < PERSONALITY_CLUSTERING["trend_strength_threshold"]
            ):
                # Very stable, no strong trends
                return "stable"
            elif (
                stability_score < (1.0 - volatility_threshold)
                and sentiment_intensity > sentiment_intensity_threshold
            ):
                # High volatility with strong emotions
                return "emotional"
            elif (
                slope > loyalty_positive_slope
                and r_squared > loyalty_r_squared
                and stability_score > DEFAULT_VALUES["stability_threshold"]
            ):
                # Consistently positive trend with moderate stability
                return "loyal"
            elif slope < skeptical_negative_slope and volatility > skeptical_volatility:
                # Negative trend with some volatility
                return "skeptical"
            elif volatility > volatility_threshold:
                # High volatility regardless of trend
                return "fluctuating"
            else:
                # Default case
                return "stable"

        except Exception as e:
            logging.warning(f"Error analyzing emotional stability for {subject}: {e}")
            return "stable"  # Default to stable on error

    @safe_db_operation
    def get_belief_context_history(
        self, subject: str, object_: str
    ) -> List[Dict[str, Any]]:
        """
        Get chronological belief change history with context for a subject-object pair.

        Args:
            subject: Subject of the beliefs
            object_: Object of the beliefs

        Returns:
            List of belief changes with context, timestamps, and sentiment scores
        """
        try:
            with self._connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, subject, predicate, object, confidence, timestamp, context
                    FROM facts 
                    WHERE subject = ? AND object = ?
                    ORDER BY timestamp ASC
                """,
                    (subject, object_),
                )

                history = []
                for row in cursor.fetchall():
                    fact_id, subj, pred, obj, confidence, timestamp, context = row

                    # Calculate sentiment score for this belief
                    from storage.memory_utils import get_sentiment_score

                    sentiment_score = get_sentiment_score(pred)

                    history.append(
                        {
                            "fact_id": fact_id,
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "confidence": confidence,
                            "timestamp": timestamp,
                            "context": context,
                            "sentiment_score": sentiment_score,
                        }
                    )

                return history

        except Exception as e:
            logging.error(f"Error getting belief context history: {e}")
            return []

    @safe_db_operation
    def forecast_sentiment(self, subject: str, object_: str) -> str:
        """
        Forecast future sentiment based on current trajectory using dynamic analysis.
        Replaces hardcoded slope thresholds with data-driven forecasting.

        Args:
            subject: Subject of the sentiment
            object_: Object of the sentiment

        Returns:
            Forecast string or empty string if no significant trend
        """
        from config.settings import (PERSONALITY_ADAPTATION,
                                     PERSONALITY_CLUSTERING)

        try:
            trajectory = self.get_sentiment_trajectory(subject, object_)

            # Check if we have enough data for forecasting
            min_facts = PERSONALITY_ADAPTATION["min_facts_for_analysis"]
            if trajectory["fact_count"] < min_facts:
                return ""

            slope = trajectory["slope"]
            recent_sentiment = trajectory["recent_sentiment"]
            volatility = trajectory["volatility"]
            r_squared = trajectory["r_squared"]

            # Dynamic forecasting based on multiple factors
            trend_strength = abs(slope) * r_squared  # How reliable the trend is
            sentiment_range = abs(recent_sentiment)  # Current sentiment intensity

            # Only forecast if trend is strong and reliable
            trend_strength_threshold = PERSONALITY_CLUSTERING[
                "trend_strength_threshold"
            ]
            if trend_strength < trend_strength_threshold:
                return ""  # No clear trend

            # Calculate confidence in forecast based on data quality
            forecast_data_quality_factor = PERSONALITY_CLUSTERING[
                "forecast_data_quality_factor"
            ]
            forecast_confidence = min(
                1.0, trajectory["fact_count"] / forecast_data_quality_factor
            )

            forecast_confidence_threshold = PERSONALITY_CLUSTERING[
                "forecast_confidence_threshold"
            ]
            if forecast_confidence < forecast_confidence_threshold:
                return ""  # Not enough confidence in forecast

            # Dynamic threshold based on volatility and trend strength
            trend_reliability_weight = PERSONALITY_ADAPTATION[
                "trend_reliability_weight"
            ]
            positive_threshold = (
                DEFAULT_VALUES["confidence_decay_threshold"] + 0.05
            ) + (volatility * trend_reliability_weight)
            negative_threshold = -0.15 - (volatility * trend_reliability_weight)

            # Forecast positive trend
            if (
                slope > positive_threshold
                and recent_sentiment < CONFIDENCE_THRESHOLDS["medium"]
            ):
                confidence_phrase = (
                    "strongly"
                    if forecast_confidence > CONFIDENCE_THRESHOLDS["medium"]
                    else "likely"
                )
                return f"Based on your recent pattern, you may {confidence_phrase} come to like {object_} more."

            # Forecast negative trend
            elif (
                slope < negative_threshold
                and recent_sentiment > -CONFIDENCE_THRESHOLDS["medium"]
            ):
                confidence_phrase = (
                    "strongly"
                    if forecast_confidence > CONFIDENCE_THRESHOLDS["medium"]
                    else "likely"
                )
                return f"Warning: you may {confidence_phrase} be growing disillusioned with {object_}."

            # Forecast stabilization
            elif (
                abs(slope) < DEFAULT_VALUES["trend_strength_threshold"]
                and volatility > VOLATILITY_THRESHOLDS["stable"]
            ):
                return f"Your feelings about {object_} appear to be stabilizing."

            # Forecast continued volatility
            elif (
                volatility > PERSONALITY_CLUSTERING["volatility_threshold"]
                and trend_strength < DEFAULT_VALUES["trend_strength_threshold"]
            ):
                return f"Your opinions about {object_} may continue to fluctuate."

            return ""

        except Exception as e:
            logging.warning(f"Error forecasting sentiment for {subject} {object_}: {e}")
            return ""

    def extract_facts(
        self, text: str, message_id: Optional[int] = None
    ) -> List[TripletFact]:
        if not isinstance(text, str):
            raise ValueError(f"Invalid input to extract_facts: {text}")
        """
        Extract facts from text and return as TripletFact objects.
        This is a wrapper around extract_triplets for backward compatibility.

        Args:
            text: Text to extract facts from
            message_id: Optional message ID for context (ignored for now)
        """
        triplets = self.extract_triplets(text)
        facts = []
        for subject, predicate, object_, confidence in triplets:
            fact = TripletFact(
                id=0,  # Will be assigned by database
                subject=subject,
                predicate=predicate,
                object=object_,
                frequency=1,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                confidence=confidence,
            )
            facts.append(fact)
        return facts

    def store_facts(self, facts: List[TripletFact]) -> List[int]:
        """
        Store a list of TripletFact objects in the database.

        Args:
            facts: List of TripletFact objects to store

        Returns:
            List of fact IDs that were stored
        """
        fact_ids = []
        for fact in facts:
            try:
                # Convert TripletFact to tuple format for store_triplets
                triplet = (fact.subject, fact.predicate, fact.object, fact.confidence)
                self.store_triplets([triplet])
                fact_ids.append(fact.id if fact.id > 0 else len(fact_ids) + 1)
            except Exception as e:
                logging.warning(f"Failed to store fact {fact}: {e}")
        return fact_ids

    @safe_db_operation
    def get_facts_about(self, subject: str) -> List[TripletFact]:
        """
        Get all facts about a specific subject.

        Args:
            subject: The subject to search for

        Returns:
            List of TripletFact objects about the subject, ranked by weighted recency + confidence
        """
        try:
            with self._connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, subject, predicate, object, confidence, timestamp, frequency
                    FROM facts 
                    WHERE subject = ?
                """,
                    (subject,),
                )

                facts = []
                for row in cursor.fetchall():
                    fact_id, subj, pred, obj, confidence, timestamp, frequency = row
                    fact = TripletFact(
                        id=fact_id,
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=confidence,
                        timestamp=timestamp,
                        frequency=frequency,
                    )
                    facts.append(fact)

                # Use weighted recency + confidence ranking instead of confidence-only sorting
                from storage.memory_utils import rank_facts
                return rank_facts(facts)

        except Exception as e:
            logging.error(f"Error getting facts about {subject}: {e}")
            return []

    @safe_db_operation
    def list_episodes(self) -> List[Dict]:
        """
        List all episodes in the database.

        Returns:
            List of episode dictionaries
        """
        try:
            with self._connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, start_time, end_time, subject_count, fact_count, summary
                    FROM episodes 
                    ORDER BY start_time DESC
                """
                )

                episodes = []
                for row in cursor.fetchall():
                    (
                        episode_id,
                        start_time,
                        end_time,
                        subject_count,
                        fact_count,
                        summary,
                    ) = row
                    episodes.append(
                        {
                            "id": episode_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "subject_count": subject_count,
                            "fact_count": fact_count,
                            "summary": summary,
                        }
                    )

                return episodes

        except Exception as e:
            logging.error(f"Error listing episodes: {e}")
            return []

    @safe_db_operation
    def show_episode(self, episode_id: int) -> Optional[Dict]:
        """
        Show details of a specific episode.

        Args:
            episode_id: ID of the episode to show

        Returns:
            Dictionary with episode details and facts, or None if not found
        """
        try:
            with self._connection_pool.get_connection() as conn:
                # Get episode details
                cursor = conn.execute(
                    """
                    SELECT id, start_time, end_time, subject_count, fact_count, summary
                    FROM episodes 
                    WHERE id = ?
                """,
                    (episode_id,),
                )

                episode_row = cursor.fetchone()
                if not episode_row:
                    return None

                episode_id, start_time, end_time, subject_count, fact_count, summary = (
                    episode_row
                )
                episode = {
                    "id": episode_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "subject_count": subject_count,
                    "fact_count": fact_count,
                    "summary": summary,
                }

                # Get facts for this episode
                cursor = conn.execute(
                    """
                    SELECT id, subject, predicate, object, confidence, timestamp
                    FROM facts 
                    WHERE episode_id = ?
                    ORDER BY timestamp ASC
                """,
                    (episode_id,),
                )

                facts = []
                for row in cursor.fetchall():
                    fact_id, subject, predicate, object_val, confidence, timestamp = row
                    fact = TripletFact(
                        id=fact_id,
                        subject=subject,
                        predicate=predicate,
                        object=object_val,
                        confidence=confidence,
                        timestamp=timestamp,
                    )
                    facts.append(fact)

                return {"episode": episode, "facts": facts}

        except Exception as e:
            logging.error(f"Error showing episode {episode_id}: {e}")
            return None

    @safe_db_operation
    def get_memory_leaders(self, top_n: int = 10) -> List[Tuple[str, int, float]]:
        """
        Get the top subjects by fact count and average confidence.

        Args:
            top_n: Number of top subjects to return

        Returns:
            List of tuples (subject, fact_count, avg_confidence)
        """
        try:
            with self._connection_pool.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT subject, COUNT(*) as fact_count, AVG(confidence) as avg_confidence
                    FROM facts 
                    GROUP BY subject
                    ORDER BY fact_count DESC, avg_confidence DESC
                    LIMIT ?
                """,
                    (top_n,),
                )

                leaders = []
                for row in cursor.fetchall():
                    subject, fact_count, avg_confidence = row
                    leaders.append((subject, fact_count, avg_confidence or 0.0))

                return leaders

        except Exception as e:
            logging.error(f"Error getting memory leaders: {e}")
            return []

    def _find_similar_facts_internal(self, triplet_text: str, conn) -> Optional[TripletFact]:
        """Internal version of find_similar_facts that uses an existing connection."""
        from scripts.embedder import embed
        import numpy as np
        from config.settings import CONFIDENCE_THRESHOLDS
        triplet_embedding = embed(triplet_text)
        if np.all(triplet_embedding == 0):
            return None
        rows = conn.execute(
            """
            SELECT id, subject, predicate, object, source_message_id, timestamp, 
                   frequency, contradiction_score, volatility_score, confidence, last_reinforced
            FROM facts
            """
        ).fetchall()
        best_match = None
        best_score = 0.0
        for row in rows:
            fact_id, subject, predicate, object_val, source_id, timestamp, frequency, contradiction_score, volatility_score, confidence, last_reinforced = row
            fact_text = f"{subject} {predicate} {object_val}"
            fact_embedding = embed(fact_text)
            if np.all(fact_embedding == 0):
                continue
            # Use proper cosine similarity instead of dot product
            similarity = self._cosine_similarity(triplet_embedding, fact_embedding)
            if similarity > best_score and similarity > CONFIDENCE_THRESHOLDS.get("high", 0.9):
                best_score = similarity
                best_match = TripletFact(
                    id=fact_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_val,
                    source_message_id=source_id,
                    timestamp=timestamp,
                    frequency=frequency,
                    contradiction_score=contradiction_score,
                    volatility_score=volatility_score,
                    confidence=confidence,
                )
        return best_match

    def _reinforce_fact_internal(self, fact_id, amount, conn):
        """Stub: No-op for reinforcement in test context."""
        return None

    @safe_db_operation
    def clear_all_facts_and_memory(self) -> int:
        """
        Clear all facts and memory entries from the database.
        
        Returns:
            Number of entries deleted
        """
        deleted_count = 0
        
        with self._connection_pool.get_connection() as conn:
            # Get counts before deletion
            facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            memory_count = conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            
            # Disable foreign key constraints temporarily for cleanup
            conn.execute("PRAGMA foreign_keys = OFF")
            
            # Clear all tables in safe order (child tables first, then parent tables)
            # Delete child tables that reference facts first
            conn.execute("DELETE FROM contradictions")
            conn.execute("DELETE FROM clusters") 
            conn.execute("DELETE FROM drift_events")
            conn.execute("DELETE FROM trust_scores")
            conn.execute("DELETE FROM fact_history")
            conn.execute("DELETE FROM summaries")
            
            # Now safe to delete parent tables
            conn.execute("DELETE FROM facts")
            conn.execute("DELETE FROM memory")
            
            # Reset episode counter
            conn.execute("DELETE FROM episodes")
            conn.execute("INSERT INTO episodes (id, start_time, subject_count, fact_count, summary) VALUES (1, CURRENT_TIMESTAMP, 0, 0, 'Fresh start')")
            
            # Re-enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            conn.commit()
            deleted_count = facts_count + memory_count
            
        logging.info(f"Cleared all memory: deleted {deleted_count} entries")
        return deleted_count

    def semantic_search(self, query: str, topk: int = 5, user_profile_id: str = None, media_type: str = None) -> list:
        # Lightweight semantic search for tests; fallback to keyword match by media_type when embeddings absent
        try:
            triplets = extract_triplets(query, message_id=-1)
        except Exception:
            triplets = []
        results = []
        # Build a simple embedding if available
        try:
            query_embedding = self.embedder.embed(query)
        except Exception:
            query_embedding = None
        sql = "SELECT subject, predicate, object, embedding, user_profile_id, media_type FROM facts WHERE 1=1"
        params = []
        if user_profile_id:
            sql += " AND user_profile_id = ?"
            params.append(user_profile_id)
        if media_type:
            sql += " AND media_type = ?"
            params.append(media_type)
        with self._connection_pool.get_connection() as conn:
            stored_facts = conn.execute(sql, params).fetchall()
        for fact in stored_facts:
            subj, pred, obj, emb, uid, mtype = fact
            similarity = 0.0
            if query_embedding is not None and emb is not None:
                try:
                    fact_embedding = eval(emb) if isinstance(emb, str) else emb
                    similarity = 1 - cosine(query_embedding, fact_embedding)
                except Exception:
                    similarity = 0.0
            # If no embedding similarity, use a simple keyword heuristic
            text_blob = f"{subj} {pred} {obj}".lower()
            if any(tok in text_blob for tok in query.lower().split()):
                similarity = max(similarity, 0.6)
            results.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "similarity": similarity,
                "user_profile_id": uid,
                "media_type": mtype
            })
        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:topk]
