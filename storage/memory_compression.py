#!/usr/bin/env python3
"""
Memory Compression Engine
Summarizes clusters of triplets into higher-level abstractions to avoid linear memory growth.
"""

import json
import logging
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import sklearn
import sklearn.cluster

from scripts.embedder import embed
from storage.db_utils import get_conn
from storage.memory_log import MemoryLog, TripletFact

# Use background logger to avoid interfering with user input
background_logger = logging.getLogger("background")


@dataclass
class TripletCluster:
    """Represents a cluster of related triplets"""

    subject: str
    triplets: List[TripletFact]
    centroid_embedding: np.ndarray
    cluster_score: float  # How cohesive the cluster is
    last_updated: str
    compressed_fact: Optional[TripletFact] = None


@dataclass
class CompressionResult:
    """Result of a compression operation"""

    original_count: int
    compressed_count: int
    compression_ratio: float
    new_facts: List[TripletFact]
    removed_facts: List[int]  # IDs of removed facts


class MemoryCompressionEngine:
    """
    Background engine that compresses memory by summarizing clusters of related facts
    """

    def __init__(
        self, memory_log: MemoryLog, compression_interval: int = 300
    ):  # 5 minutes
        self.memory_log = memory_log
        self.compression_interval = compression_interval
        self.running = False
        self.thread = None
        self.last_compression_time = 0

        # Configuration
        self.min_cluster_size = 3  # Minimum facts to form a cluster
        self.similarity_threshold = 0.7  # Minimum similarity for clustering
        self.compression_threshold = 0.8  # Minimum cluster score to compress
        self.max_facts_per_cluster = 10  # Maximum facts to include in compression

    def start_background_loop(self):
        """Start the background compression loop (now handled by Celery)"""
        # Background tasks are now handled by Celery task queue
        # This method is kept for backward compatibility
        background_logger.info("🗜️ Memory compression now handled by Celery task queue")

    def stop_background_loop(self):
        """Stop the background compression loop (now handled by Celery)"""
        # Background tasks are now handled by Celery task queue
        # This method is kept for backward compatibility
        background_logger.info("🛑 Memory compression now handled by Celery task queue")

    def _background_loop(self):
        """Main background loop for memory compression (deprecated)"""
        # This method is deprecated - use Celery tasks instead
        background_logger.warning("_background_loop is deprecated - use Celery tasks")
        pass

    def trigger_compression(self):
        """Manually trigger memory compression"""
        self._check_and_compress_memory()

    def _check_and_compress_memory(self):
        """Check if compression is needed and perform it"""
        background_logger.info("🗜️ Checking for memory compression opportunities...")

        # New pruning logic
        old_low_conf_facts = self.memory_log.get_old_low_confidence_facts(
            days=30, conf_threshold=0.1
        )
        for fact in old_low_conf_facts:
            self.memory_log.delete_fact(fact.id)

        # Get all facts
        facts = self.memory_log.get_all_facts(prune_contradictions=False)

        if len(facts) < self.min_cluster_size * 2:
            background_logger.info("   Not enough facts for compression")
            return

        # Check for large clusters that need compression
        compressed_count = self._check_cluster_size_and_compress()

        # Find clusters
        clusters = self._find_clusters(facts)

        if not clusters:
            background_logger.info("   No suitable clusters found for compression")
            return

        # Compress high-scoring clusters
        for cluster in clusters:
            if cluster.cluster_score >= self.compression_threshold:
                result = self._compress_cluster(cluster)
                if result:
                    compressed_count += 1
                    background_logger.info(
                        f"   Compressed cluster '{cluster.subject}': {result.original_count} → {result.compressed_count} facts"
                    )

        if compressed_count > 0:
            background_logger.info(f"✅ Compressed {compressed_count} clusters")
        else:
            background_logger.info("   No clusters met compression threshold")

    def _check_cluster_size_and_compress(self) -> int:
        """Check cluster sizes and compress those that exceed threshold."""
        compressed_count = 0

        with get_conn(self.memory_log.db_path) as conn:
            # Get clusters that exceed size threshold
            rows = conn.execute(
                """
                SELECT id, subject, fact_ids, cluster_size 
                FROM clusters 
                WHERE cluster_size > ?
                ORDER BY cluster_size DESC
            """,
                (self.max_facts_per_cluster,),
            ).fetchall()

            for row in rows:
                cluster_id, subject, fact_ids_json, cluster_size = row
                fact_ids = json.loads(fact_ids_json) if fact_ids_json else []

                if len(fact_ids) > self.max_facts_per_cluster:
                    result = self._compress_cluster_by_id(
                        cluster_id, subject, fact_ids, conn
                    )
                    if result:
                        compressed_count += 1
                        background_logger.info(
                            f"   Compressed large cluster '{subject}' (size {cluster_size}): {result.original_count} → {result.compressed_count} facts"
                        )

        return compressed_count

    def _compress_cluster_by_id(
        self, cluster_id: int, subject: str, fact_ids: List[int], conn
    ) -> Optional[CompressionResult]:
        """Compress a specific cluster by ID."""
        try:
            # Get the facts for this cluster
            placeholders = ",".join(["?" for _ in fact_ids])
            fact_rows = conn.execute(
                f"""
                SELECT id, subject, predicate, object, confidence, contradiction_score, volatility_score
                FROM facts 
                WHERE id IN ({placeholders})
                ORDER BY confidence DESC, timestamp DESC
            """,
                fact_ids,
            ).fetchall()

            if len(fact_rows) < self.min_cluster_size:
                return None

            # Convert to TripletFact objects
            facts = []
            for row in fact_rows:
                (
                    fact_id,
                    subj,
                    pred,
                    obj,
                    confidence,
                    contradiction_score,
                    volatility_score,
                ) = row
                fact = TripletFact(
                    id=fact_id,
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    source_message_id=0,
                    timestamp="",
                    frequency=1,
                    contradiction_score=contradiction_score,
                    volatility_score=volatility_score,
                )
                fact.decayed_confidence = confidence
                facts.append(fact)

            # Create cluster object
            cluster = TripletCluster(
                subject=subject,
                triplets=facts,
                centroid_embedding=np.zeros(
                    384
                ),  # Will be calculated during compression
                cluster_score=0.8,  # Assume good cluster if it's large
                last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Compress the cluster
            result = self._compress_cluster(cluster)

            if result:
                # Update the cluster in database
                new_fact_ids = [fact.id for fact in result.new_facts]
                new_fact_ids_json = json.dumps(new_fact_ids)

                # Calculate new centroid
                embeddings = []
                for fact in result.new_facts:
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
                    embedding = embed(fact_text)
                    if np.all(embedding == 0):
                        background_logger.warning(
                            f"Warning: Embedding failed for fact_text '{fact_text}' in _compress_cluster_by_id."
                        )
                    embeddings.append(embedding)

                if embeddings:
                    new_centroid = np.mean(embeddings, axis=0)
                    conn.execute(
                        """
                        UPDATE clusters 
                        SET embedding=?, fact_ids=?, cluster_size=?, timestamp=CURRENT_TIMESTAMP 
                        WHERE id=?
                    """,
                        (
                            new_centroid.tobytes(),
                            new_fact_ids_json,
                            len(new_fact_ids),
                            cluster_id,
                        ),
                    )

                # Delete the original facts
                for fact_id in result.removed_facts:
                    conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))

                conn.commit()

            return result

        except Exception as e:
            background_logger.exception(f"Error compressing cluster {cluster_id}: {e}")
            return None

    def compress_cluster(self, subject: str) -> Optional[CompressionResult]:
        """Manually compress a specific subject's cluster."""
        with get_conn(self.memory_log.db_path) as conn:
            # Get the cluster for this subject
            row = conn.execute(
                """
                SELECT id, fact_ids, cluster_size 
                FROM clusters 
                WHERE subject=? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """,
                (subject,),
            ).fetchone()

            if not row:
                background_logger.warning(f"No cluster found for subject '{subject}'")
                return None

            cluster_id, fact_ids_json, cluster_size = row
            fact_ids = json.loads(fact_ids_json) if fact_ids_json else []

            if len(fact_ids) < self.min_cluster_size:
                background_logger.warning(
                    f"Cluster for '{subject}' too small ({len(fact_ids)} facts, need {self.min_cluster_size})"
                )
                return None

            return self._compress_cluster_by_id(cluster_id, subject, fact_ids, conn)

    def _find_clusters(self, facts: List[TripletFact]) -> List[TripletCluster]:
        """Find clusters of related facts by subject, using MiniBatchKMeans for large groups."""
        subject_groups = defaultdict(list)
        for fact in facts:
            subject_groups[fact.subject].append(fact)

        clusters = []

        for subject, subject_facts in subject_groups.items():
            if len(subject_facts) < self.min_cluster_size:
                continue

            # Embed facts
            embeddings = []
            for fact in subject_facts:
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
                embedding = embed(fact_text)
                if np.all(embedding == 0):
                    continue
                embeddings.append(embedding)

            if len(embeddings) < self.min_cluster_size:
                continue
            embeddings = np.array(embeddings)

            # If large group, use KMeans to sub-cluster
            if len(subject_facts) > 100:
                n_clusters = max(1, len(subject_facts) // 50)  # Approximate
                kmeans = sklearn.cluster.MiniBatchKMeans(
                    n_clusters=n_clusters, random_state=42
                )
                labels = kmeans.fit_predict(embeddings)

                for label in range(n_clusters):
                    idx = np.where(labels == label)[0]
                    cluster_facts = [subject_facts[i] for i in idx]
                    cluster_embs = embeddings[idx]
                    if len(cluster_facts) < self.min_cluster_size:
                        continue
                    centroid = np.mean(cluster_embs, axis=0)
                    cohesion_scores = [
                        self._cosine_similarity(centroid, emb) for emb in cluster_embs
                    ]
                    cluster_score = np.mean(cohesion_scores)
                    if cluster_score >= self.similarity_threshold:
                        clusters.append(
                            TripletCluster(
                                subject=subject,
                                triplets=cluster_facts,
                                centroid_embedding=centroid,
                                cluster_score=cluster_score,
                                last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                            )
                        )
            else:
                # Calculate cluster centroid
                embeddings = []
                for fact in subject_facts:
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
                    embedding = embed(fact_text)
                    if np.all(embedding == 0):
                        background_logger.warning(
                            f"Warning: Embedding failed for fact_text '{fact_text}' in _find_clusters."
                        )
                    embeddings.append(embedding)

                if len(embeddings) < self.min_cluster_size:
                    continue

                # Calculate centroid
                centroid = np.mean(embeddings, axis=0)

                # Calculate cluster cohesion score
                cohesion_scores = []
                for embedding in embeddings:
                    similarity = self._cosine_similarity(centroid, embedding)
                    cohesion_scores.append(similarity)

                cluster_score = np.mean(cohesion_scores)

                # Only include clusters with good cohesion
                if cluster_score >= self.similarity_threshold:
                    clusters.append(
                        TripletCluster(
                            subject=subject,
                            triplets=subject_facts,
                            centroid_embedding=centroid,
                            cluster_score=cluster_score,
                            last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )

        # Sort by cluster score (highest first)
        clusters.sort(key=lambda c: c.cluster_score, reverse=True)
        return clusters

    def _compress_cluster(self, cluster: TripletCluster) -> Optional[CompressionResult]:
        """Compress a cluster of facts into a single summarized fact"""
        try:
            # Prepare facts for LLM summarization
            fact_texts = []
            for fact in cluster.triplets[: self.max_facts_per_cluster]:
                fact_texts.append(f"{fact.subject} {fact.predicate} {fact.object}")

            # Create summarization prompt
            prompt = f"""Summarize these related facts about '{cluster.subject}' into one general fact in JSON triplet format.

Facts:
{chr(10).join(f"- {text}" for text in fact_texts)}

Return only a JSON object: {{"subject": "...", "predicate": "...", "object": "..."}}

The summary should capture the overall relationship or pattern between these facts:"""

            # Call LLM for summarization
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()

            # Parse response
            match = re.search(r"\{.*?\}", response.json()["response"], re.DOTALL)
            if not match:
                return None

            summary_data = json.loads(match.group(0))

            if not all(k in summary_data for k in ("subject", "predicate", "object")):
                return None

            # Create compressed fact
            compressed_fact = TripletFact(
                id=0,  # Will be assigned by database
                subject=summary_data["subject"],
                predicate=summary_data["predicate"],
                object=summary_data["object"],
                source_message_id=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                frequency=len(cluster.triplets),
                contradiction_score=0.0,
                volatility_score=cluster.cluster_score,  # Use cluster score as volatility
            )

            # Store compressed fact and remove original facts
            with get_conn(self.memory_log.db_path) as conn:
                # Insert compressed fact
                cursor = conn.execute(
                    "INSERT INTO facts (subject, predicate, object, source_message_id, timestamp, frequency, contradiction_score, volatility_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        compressed_fact.subject,
                        compressed_fact.predicate,
                        compressed_fact.object,
                        compressed_fact.source_message_id,
                        compressed_fact.timestamp,
                        compressed_fact.frequency,
                        compressed_fact.contradiction_score,
                        compressed_fact.volatility_score,
                    ),
                )
                compressed_fact.id = cursor.lastrowid

                # Remove original facts
                original_ids = [fact.id for fact in cluster.triplets]
                conn.execute(
                    "DELETE FROM facts WHERE id IN ({})".format(
                        ",".join("?" * len(original_ids))
                    ),
                    original_ids,
                )

                # Log compression event
                conn.execute(
                    "INSERT INTO compression_log (compressed_fact_id, original_fact_ids, cluster_score, timestamp) VALUES (?, ?, ?, ?)",
                    (
                        compressed_fact.id,
                        ",".join(map(str, original_ids)),
                        cluster.cluster_score,
                        compressed_fact.timestamp,
                    ),
                )

                conn.commit()

            return CompressionResult(
                original_count=len(cluster.triplets),
                compressed_count=1,
                compression_ratio=1.0 / len(cluster.triplets),
                new_facts=[compressed_fact],
                removed_facts=original_ids,
            )

        except Exception as e:
            background_logger.exception("❌ Cluster compression failed")
            return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_compression_stats(self) -> Dict:
        """Get compression statistics"""
        with get_conn(self.memory_log.db_path) as conn:
            # Check if compression_log table exists
            try:
                compression_count = conn.execute(
                    "SELECT COUNT(*) FROM compression_log"
                ).fetchone()[0]
                total_compressed = (
                    conn.execute(
                        "SELECT SUM(original_fact_ids) FROM compression_log"
                    ).fetchone()[0]
                    or 0
                )
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                compression_count = 0
                total_compressed = 0

            current_facts = len(self.memory_log.get_all_facts())

            return {
                "compression_events": compression_count,
                "total_facts_compressed": total_compressed,
                "current_facts": current_facts,
                "compression_ratio": (
                    total_compressed / (total_compressed + current_facts)
                    if (total_compressed + current_facts) > 0
                    else 0.0
                ),
            }

    def create_compression_log_table(self):
        """Create the compression log table if it doesn't exist"""
        with get_conn(self.memory_log.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compression_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    compressed_fact_id INTEGER,
                    original_fact_ids TEXT,
                    cluster_score REAL,
                    timestamp TEXT,
                    FOREIGN KEY (compressed_fact_id) REFERENCES facts (id)
                )
            """
            )
            conn.commit()


def benchmark_compression(memory_log, n_facts=100000):
    """Benchmark compression performance with simulated facts."""
    start = time.time()
    # Simulate facts
    simulated_facts = [
        TripletFact(
            i,
            f"subject_{i%100}",
            "predicate",
            "object",
            frequency=1,
            timestamp="2024-01-01",
        )
        for i in range(n_facts)
    ]
    engine = MemoryCompressionEngine(memory_log)
    clusters = engine._find_clusters(simulated_facts)
    end = time.time()
    latency = end - start
    print(
        f"Benchmark: Clustered {n_facts} facts in {latency:.2f} seconds, found {len(clusters)} clusters"
    )
    return latency
