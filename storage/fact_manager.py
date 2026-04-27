"""
Enhanced fact management for MeRNSTA that properly handles fact storage,
contradiction detection, and user profile scoping.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from storage.memory_log import MemoryLog, TripletFact
from storage.memory_utils import detect_contradictions, calculate_contradiction_score
from config.environment import get_settings

class FactManager:
    """
    Manages fact storage with enhanced contradiction detection and user profiling.
    """
    
    def __init__(self, memory_log: MemoryLog):
        self.memory_log = memory_log
        self.settings = get_settings()
        
    def store_facts_with_validation(self, triplets: List[Tuple], user_profile_id: str = "default_user", 
                                  session_id: str = None) -> Dict[str, Any]:
        """
        Store facts with validation, contradiction detection, and proper update handling.
        
        Args:
            triplets: List of (subject, predicate, object, confidence) tuples
            user_profile_id: User profile ID
            session_id: Current session ID
            
        Returns:
            Dictionary with storage results
        """
        
        stored_ids = []
        contradictions = []
        summary_messages = []
        
        # Get existing facts for this user
        all_facts = self.memory_log.get_all_facts()
        user_facts = [f for f in all_facts if hasattr(f, 'user_profile_id') and 
                     getattr(f, 'user_profile_id', None) == user_profile_id]
        
        if not user_facts:
            # Fallback to subject-based filtering
            user_facts = [f for f in all_facts if f.subject.lower() in ['user', 'i', 'me']]
        
        for triplet in triplets:
            try:
                # Parse triplet format
                if len(triplet) < 3:
                    continue
                    
                subject, predicate, object_ = triplet[:3]
                
                # Handle enhanced format with metadata
                if len(triplet) == 4 and isinstance(triplet[3], dict):
                    meta = triplet[3]
                    confidence = meta.get("confidence", 1.0)
                    context = meta.get("context", None)
                    media_type = meta.get("media_type", "text")
                    media_data = meta.get("media_data", None)
                else:
                    confidence = triplet[3] if len(triplet) > 3 else 1.0
                    context = triplet[4] if len(triplet) > 4 else None
                    media_type = triplet[5] if len(triplet) > 5 else "text"
                    media_data = triplet[6] if len(triplet) > 6 else None
                
                # Normalize subject to "user" for consistency
                if subject.lower() in ['i', 'me', 'my', 'myself']:
                    subject = 'user'
                elif subject.lower() in ['actually', 'really', 'definitely', 'honestly']:
                    # These are adverbs that got misidentified as subjects
                    subject = 'user'
                
                # Normalize predicates for better contradiction detection
                if predicate.lower() in ['dislike', 'hate', 'despise']:
                    predicate = 'hate'  # Use "hate" as canonical form for stronger contradiction signal
                elif predicate.lower() in ['like', 'love', 'enjoy', 'prefer']:
                    predicate = 'like'  # Use "like" as canonical form
                
                # Check for temporal indicators in the text
                is_update = False
                original_object = object_
                
                # Use linguistic analysis to detect temporal modifiers
                try:
                    from storage.spacy_extractor import nlp
                    if nlp:
                        doc = nlp(original_object)
                        # Check for temporal expressions using spaCy's entity recognition
                        for ent in doc.ents:
                            if ent.label_ in ['DATE', 'TIME']:
                                is_update = True
                                break
                        # Check for temporal adverbs
                        for token in doc:
                            if token.pos_ == 'ADV' and token.dep_ in ['advmod', 'npadvmod']:
                                if any(temp in token.text.lower() for temp in ['now', 'currently', 'anymore']):
                                    is_update = True
                                    break
                except:
                    # Fallback: basic temporal word detection
                    if any(word in original_object.lower().split() for word in ['now', 'anymore', 'currently']):
                        is_update = True
                
                # Normalize objects by removing temporal suffixes
                object_normalized = object_
                # Use regex to remove common temporal endings
                import re
                temporal_pattern = r'\s+(now|today|currently|right now|these days|anymore)$'
                match = re.search(temporal_pattern, object_normalized, re.IGNORECASE)
                if match:
                    object_normalized = object_normalized[:match.start()].strip()
                    is_update = True
                
                # Check for contradictions before storing (using normalized object)
                new_fact_tuple = (subject, predicate, object_normalized)
                detected_contradictions = self._check_contradictions(new_fact_tuple, user_facts)
                
                # Track facts to delete
                facts_to_delete = []
                
                if detected_contradictions:
                    # Log contradiction but still store the fact
                    for existing_fact, contradiction_score in detected_contradictions:
                        contradiction_info = {
                            "new_fact": f"{subject} {predicate} {object_}",
                            "existing_fact": f"{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}",
                            "score": contradiction_score,
                            "existing_fact_id": existing_fact.id
                        }
                        contradictions.append(contradiction_info)
                        
                        # If this is an update (temporal modifier present), mark old fact for deletion
                        if is_update and contradiction_score > 0.7:
                            facts_to_delete.append(existing_fact.id)
                            summary_messages.append(f"Updated fact: replaced '{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}' with new information")
                        else:
                            summary_messages.append(
                                f"⚠️ This contradicts your earlier statement: '{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}'. Which is correct?"
                            )
                
                # Store the fact with proper metadata
                with self.memory_log._connection_pool.get_connection() as conn:
                    # Delete old facts if this is an update
                    for fact_id in facts_to_delete:
                        conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                        # Remove from user_facts list to avoid further processing
                        user_facts = [f for f in user_facts if f.id != fact_id]
                    
                    # Check if similar fact already exists
                    similar_fact = self._find_similar_fact(subject, predicate, object_, conn)
                    
                    if similar_fact and not is_update:
                        # Reinforce existing fact
                        self._reinforce_fact(similar_fact.id, 0.1, conn)
                        stored_ids.append(similar_fact.id)
                        summary_messages.append(f"Reinforced existing fact: {subject} {predicate} {object_}")
                    else:
                        # Store new fact
                        fact_id = self._store_new_fact(
                            subject, predicate, object_, confidence, 
                            context, media_type, media_data,
                            session_id, user_profile_id, conn
                        )
                        stored_ids.append(fact_id)
                        
                        # Log contradictions in database if any were detected
                        for contradiction in detected_contradictions:
                            existing_fact, score = contradiction
                            self._log_contradiction(
                                fact_id, existing_fact.id,
                                f"{subject} {predicate} {object_}",
                                f"{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}",
                                score, conn
                            )
                        
                        summary_messages.append(f"Stored new fact: {subject} {predicate} {object_}")
                
            except Exception as e:
                logging.error(f"Error storing fact {triplet}: {e}")
                summary_messages.append(f"❌ Failed to store fact: {e}")
        
        # After storing all facts, mark volatility for contradictory facts
        volatile_topics = []
        try:
            volatile_topics = self._mark_volatility_for_contradictory_facts(user_profile_id, session_id)
            if volatile_topics:
                for subject, obj in volatile_topics:
                    summary_messages.append(f"🔥 Volatile topic detected: {subject} has conflicting feelings about {obj}")
                    # Also add a clear volatility indicator to the main summary
                    summary_messages.append(f"⚠️ You've changed your mind multiple times about {obj} - this topic seems volatile")
        except Exception as e:
            logging.warning(f"Failed to mark volatility: {e}")
        
        return {
            "stored_ids": stored_ids,
            "contradictions": contradictions,
            "summary_messages": summary_messages,
            "volatile_topics": volatile_topics,
            "success": len(stored_ids) > 0
        }
    
    def _check_contradictions(self, new_fact: Tuple[str, str, str], 
                            existing_facts: List[TripletFact]) -> List[Tuple[TripletFact, float]]:
        """Check for contradictions between new fact and existing facts using enhanced semantic analysis."""
        
        subject, predicate, object_ = new_fact
        contradictions = []
        
        # Normalize the new fact's object for comparison
        def normalize_object(obj):
            temporal_suffixes = [' now', ' today', ' currently', ' right now', ' these days', ' anymore']
            for suffix in temporal_suffixes:
                if obj.lower().endswith(suffix):
                    return obj[:-len(suffix)].strip()
            return obj
        
        normalized_new_object = normalize_object(object_)
        
        # Enhanced semantic similarity check using spaCy if available
        def objects_similar(obj1: str, obj2: str) -> bool:
            """Check if two objects are semantically similar."""
            obj1_norm = obj1.lower().strip()
            obj2_norm = obj2.lower().strip()
            
            # Exact or substring match
            if obj1_norm == obj2_norm or obj1_norm in obj2_norm or obj2_norm in obj1_norm:
                return True
                
            # Try spaCy similarity
            try:
                from storage.spacy_extractor import nlp
                if nlp:
                    doc1 = nlp(obj1_norm)
                    doc2 = nlp(obj2_norm)
                    similarity = doc1.similarity(doc2)
                    return similarity > 0.85
            except:
                pass
                
            # Fallback to word overlap
            words1 = set(obj1_norm.split())
            words2 = set(obj2_norm.split())
            if len(words1) == 0 or len(words2) == 0:
                return False
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = overlap / union if union > 0 else 0.0
            return similarity > 0.7
        
        # Enhanced predicate conflict detection
        def predicates_conflict(pred1: str, pred2: str) -> bool:
            """Check if two predicates are contradictory."""
            pred1_norm = pred1.lower().strip()
            pred2_norm = pred2.lower().strip()
            
            # Handle like/dislike and other common antonyms
            positive_predicates = {'like', 'love', 'enjoy', 'prefer', 'want'}
            negative_predicates = {'dislike', 'hate', 'despise', 'avoid', 'detest'}
            
            if ((pred1_norm in positive_predicates and pred2_norm in negative_predicates) or
                (pred1_norm in negative_predicates and pred2_norm in positive_predicates)):
                return True
                
            # Check for direct negation patterns
            if (pred1_norm.startswith('not_') and pred1_norm[4:] == pred2_norm) or \
               (pred2_norm.startswith('not_') and pred2_norm[4:] == pred1_norm):
                return True
                
            return False
        
        # Enhanced preference conflict detection
        def preference_conflict(pred1: str, obj1: str, pred2: str, obj2: str) -> bool:
            """
            Dynamic preference conflict detection using semantic similarity and NLP.
            No hardcoded categories - adapts based on semantic understanding.
            """
            pred1_norm = pred1.lower().strip()
            pred2_norm = pred2.lower().strip()
            
            # Both must be preference predicates
            preference_predicates = {'prefer', 'like', 'love', 'enjoy', 'want', 'choose'}
            if pred1_norm not in preference_predicates or pred2_norm not in preference_predicates:
                return False
            
            obj1_norm = obj1.lower().strip()
            obj2_norm = obj2.lower().strip()
            
            if obj1_norm == obj2_norm:
                return False  # Same object, not conflicting
            
            # DYNAMIC APPROACH 1: Use spaCy semantic similarity
            try:
                import spacy
                from config.settings import get_config
                config = get_config()
                spacy_model = config.get('spacy', {}).get('model', 'en_core_web_sm')
                nlp = spacy.load(spacy_model)
                doc1 = nlp(obj1_norm)
                doc2 = nlp(obj2_norm)
                
                # If objects are semantically related but different, check for conflict
                similarity = doc1.similarity(doc2)
                if 0.3 < similarity < 0.9:  # Related but not identical
                    print(f"[PreferenceConflict] Semantic similarity: '{obj1_norm}' vs '{obj2_norm}' = {similarity:.3f}")
                    return True
            except Exception as e:
                print(f"[PreferenceConflict] spaCy similarity failed, using fallback: {e}")
            
            # DYNAMIC APPROACH 2: Simple preference conflicts
            if pred1_norm == pred2_norm == "prefer":
                # For "prefer X" vs "prefer Y", if both are simple objects, likely conflicting
                obj1_words = obj1_norm.split()
                obj2_words = obj2_norm.split()
                
                if (len(obj1_words) <= 2 and len(obj2_words) <= 2 and 
                    obj1_norm != obj2_norm):
                    print(f"[PreferenceConflict] Simple preference conflict: '{pred1_norm} {obj1_norm}' vs '{pred2_norm} {obj2_norm}'")
                    return True
            
            # DYNAMIC APPROACH 3: Memory-based learning (future enhancement)
            # The system could learn conflict patterns from past contradictions
            # and become more intelligent over time
            
            return False
        
        for existing_fact in existing_facts:
            # Only check facts with same subject
            if existing_fact.subject.lower() != subject.lower():
                continue
            
            # Normalize existing fact's object for comparison
            normalized_existing_object = normalize_object(existing_fact.object)
            
            # Check if objects are semantically similar
            if objects_similar(normalized_new_object, normalized_existing_object):
                # Check if predicates conflict
                if predicates_conflict(predicate, existing_fact.predicate):
                    score = 0.9  # High contradiction score for semantic conflicts
                    contradictions.append((existing_fact, score))
                    print(f"[ContradictionDetection] Found conflict: '{subject} {predicate} {object_}' vs '{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}' (score: {score})")
                    continue
            
            # Check for preference conflicts (e.g., "prefer tea" vs "prefer coffee")
            if preference_conflict(predicate, normalized_new_object, existing_fact.predicate, normalized_existing_object):
                score = 0.8  # High score for preference conflicts
                contradictions.append((existing_fact, score))
                print(f"[ContradictionDetection] Found preference conflict: '{subject} {predicate} {object_}' vs '{existing_fact.subject} {existing_fact.predicate} {existing_fact.object}' (score: {score})")
                continue
            
            # Check for same predicate with different objects (potential contradiction)
            if (existing_fact.predicate.lower() == predicate.lower() and 
                normalized_existing_object.lower() != normalized_new_object.lower()):
                # Use semantic similarity to determine if this is a contradiction
                score = calculate_contradiction_score(
                    TripletFact(0, subject, predicate, object_, 1, time.time(), 1),
                    existing_fact
                )
                if score > 0.5:  # Threshold for contradiction
                    contradictions.append((existing_fact, score))
        
        return contradictions
    
    def _mark_volatility_for_contradictory_facts(self, user_profile_id: str, session_id: str) -> List[Tuple[str, str]]:
        """Mark facts as volatile when 2+ contradictory facts exist for same (subject, object) pair."""
        
        # Get all facts for this user/session
        all_facts = self.memory_log.get_all_facts()
        user_facts = [f for f in all_facts if hasattr(f, 'user_profile_id') and 
                     getattr(f, 'user_profile_id', None) == user_profile_id]
        
        if not user_facts:
            # Fallback to subject-based filtering
            user_facts = [f for f in all_facts if f.subject.lower() in ['user', 'i', 'me']]
        
        # Group facts by (subject, object) pairs using semantic similarity
        from collections import defaultdict
        topic_groups = defaultdict(list)
        volatile_topics_found = []
        
        def objects_similar(obj1: str, obj2: str) -> bool:
            """Check if two objects are semantically similar."""
            obj1_norm = obj1.lower().strip()
            obj2_norm = obj2.lower().strip()
            
            if obj1_norm == obj2_norm or obj1_norm in obj2_norm or obj2_norm in obj1_norm:
                return True
                
            try:
                from storage.spacy_extractor import nlp
                if nlp:
                    doc1 = nlp(obj1_norm)
                    doc2 = nlp(obj2_norm)
                    return doc1.similarity(doc2) > 0.85
            except:
                pass
                
            words1 = set(obj1_norm.split())
            words2 = set(obj2_norm.split())
            if len(words1) == 0 or len(words2) == 0:
                return False
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return (overlap / union if union > 0 else 0.0) > 0.7
        
        # Group facts by similar (subject, object) pairs
        for fact in user_facts:
            group_key = None
            for existing_key in topic_groups.keys():
                existing_subj, existing_obj = existing_key
                if (fact.subject.lower() == existing_subj.lower() and 
                    objects_similar(fact.object, existing_obj)):
                    group_key = existing_key
                    break
            
            if group_key is None:
                group_key = (fact.subject, fact.object)
            
            topic_groups[group_key].append(fact)
        
        # Check each group for contradictions and mark as volatile if needed
        for (subject, obj), group_facts in topic_groups.items():
            contradictory_facts = []
            
            # Find contradictory predicates within the group
            predicates = [f.predicate.lower() for f in group_facts]
            positive_predicates = {'like', 'love', 'enjoy', 'prefer', 'want'}
            negative_predicates = {'dislike', 'hate', 'despise', 'avoid', 'detest'}
            
            has_positive = any(p in positive_predicates for p in predicates)
            has_negative = any(p in negative_predicates for p in predicates)
            
            # ENHANCED: More sensitive volatility detection
            if has_positive and has_negative:
                contradictory_facts = group_facts
            elif len(group_facts) >= 3:
                # If we have 3+ facts about the same thing, it might be volatile even without direct contradictions
                contradictory_facts = group_facts
            elif len(set(predicates)) >= 2 and len(group_facts) >= 2:
                # Different predicates for the same object = potential volatility
                contradictory_facts = group_facts
            
            # If 2+ facts exist that could be volatile, mark all as volatile
            if len(contradictory_facts) >= 2:
                # DYNAMIC VOLATILITY: Calculate based on actual contradiction patterns
                # More contradictions = higher volatility score
                base_volatility = len(contradictory_facts) / 5.0  # Scale factor
                pattern_bonus = 0.2 if len(set(f.predicate for f in contradictory_facts)) > 2 else 0.0
                volatility_score = min(1.0, base_volatility + pattern_bonus)
                
                print(f"[VolatilityMarking] Marking {len(contradictory_facts)} facts as volatile for topic: {subject} -> {obj}")
                print(f"[VolatilityMarking] Dynamic volatility score: {volatility_score:.3f}")
                volatile_topics_found.append((subject, obj))
                
                # Update facts in database to mark as volatile
                with self.memory_log._connection_pool.get_connection() as conn:
                    for fact in contradictory_facts:
                        try:
                            conn.execute(
                                """UPDATE facts 
                                   SET volatility_score = ?, contradiction_score = 1.0 
                                   WHERE id = ?""",
                                (volatility_score, fact.id)
                            )
                        except Exception as e:
                            logging.warning(f"Failed to update volatility for fact {fact.id}: {e}")
                    conn.commit()
        
        return volatile_topics_found
    
    def _are_contradictory_predicates(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory using semantic analysis."""
        
        pred1_lower = pred1.lower().strip()
        pred2_lower = pred2.lower().strip()
        
        # First try to use LLM for semantic contradiction detection
        try:
            import requests
            prompt = f"Are these two predicates contradictory? '{pred1_lower}' and '{pred2_lower}'. Answer only 'yes' or 'no'."
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=5
            )
            if response.status_code == 200:
                answer = response.json()["response"].strip().lower()
                return answer == "yes"
        except:
            pass
        
        # Fallback to embedding similarity
        try:
            from scripts.embedder import embed
            import numpy as np
            
            emb1 = embed(pred1_lower)
            emb2 = embed(pred2_lower)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # If similarity is very low (opposite meanings), they might be contradictory
            # But we need to be careful - "like" and "eat" are different but not contradictory
            # Only consider them contradictory if they're somewhat related but opposite
            if similarity < -0.3:  # Negative similarity suggests opposition
                return True
            
            # Check if one contains negation of the other
            if (f"not {pred1_lower}" in pred2_lower or f"don't {pred1_lower}" in pred2_lower or
                f"not {pred2_lower}" in pred1_lower or f"don't {pred2_lower}" in pred1_lower):
                return True
                
        except:
            pass
        
        # Final fallback: basic negation check
        negations = ['not', "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't", "aren't"]
        for neg in negations:
            if (neg in pred1_lower and pred2_lower.replace(neg, '').strip() == pred1_lower.replace(neg, '').strip()) or \
               (neg in pred2_lower and pred1_lower.replace(neg, '').strip() == pred2_lower.replace(neg, '').strip()):
                return True
        
        return False
    
    def _find_similar_fact(self, subject: str, predicate: str, object_: str, conn) -> Optional[TripletFact]:
        """Find if a similar fact already exists."""
        
        # Look for exact match first
        cursor = conn.execute(
            """
            SELECT id, subject, predicate, object, source_message_id, timestamp, 
                   frequency, contradiction_score, volatility_score, confidence
            FROM facts 
            WHERE LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?) AND LOWER(object) = LOWER(?)
            """,
            (subject, predicate, object_)
        )
        
        row = cursor.fetchone()
        if row:
            return TripletFact(*row)
        
        return None
    
    def _store_new_fact(self, subject: str, predicate: str, object_: str, confidence: float,
                       context: Any, media_type: str, media_data: Any,
                       session_id: str, user_profile_id: str, conn) -> int:
        """Store a new fact in the database."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        episode_id = 1  # Default episode
        
        import json
        
        cursor = conn.execute(
            """
            INSERT INTO facts (subject, predicate, object, confidence, timestamp, episode_id, 
                             frequency, context, media_type, media_data, session_id, user_profile_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject, predicate, object_, confidence, timestamp, episode_id, 1,
                json.dumps(context) if context else None,
                media_type if media_type else "text",
                media_data, session_id, user_profile_id
            )
        )
        
        conn.commit()
        return cursor.lastrowid
    
    def _reinforce_fact(self, fact_id: int, boost: float, conn) -> None:
        """Reinforce an existing fact by updating its confidence and frequency."""
        
        # Get current values
        cursor = conn.execute(
            "SELECT confidence, frequency FROM facts WHERE id = ?",
            (fact_id,)
        )
        row = cursor.fetchone()
        
        if row:
            current_confidence, current_frequency = row
            new_confidence = min(1.0, current_confidence + boost)
            new_frequency = current_frequency + 1
            
            conn.execute(
                """
                UPDATE facts 
                SET confidence = ?, frequency = ?, last_reinforced = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (new_confidence, new_frequency, fact_id)
            )
            conn.commit()
    
    def _log_contradiction(self, fact_a_id: int, fact_b_id: int, fact_a_text: str, 
                          fact_b_text: str, confidence: float, conn) -> None:
        """Log a contradiction in the contradictions table."""
        
        cursor = conn.execute(
            """
            INSERT INTO contradictions (fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (fact_a_id, fact_b_id, fact_a_text, fact_b_text, confidence)
        )
        conn.commit()
        
        logging.info(f"Logged contradiction: '{fact_a_text}' vs '{fact_b_text}' (confidence: {confidence:.2f})")
    
    def get_contradictions(self, user_profile_id: str = None, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get contradictions, optionally filtered by user and resolution status."""
        
        try:
            with self.memory_log._connection_pool.get_connection() as conn:
                query = """
                    SELECT c.id, c.fact_a_id, c.fact_b_id, c.fact_a_text, c.fact_b_text, 
                           c.confidence, c.resolved, c.timestamp
                    FROM contradictions c
                """
                params = []
                
                # Add filters
                where_clauses = []
                if resolved is not None:
                    where_clauses.append("c.resolved = ?")
                    params.append(resolved)
                
                if user_profile_id:
                    # Join with facts to filter by user_profile_id
                    query += """
                        JOIN facts f_a ON c.fact_a_id = f_a.id
                        JOIN facts f_b ON c.fact_b_id = f_b.id
                    """
                    where_clauses.append("(f_a.user_profile_id = ? OR f_b.user_profile_id = ?)")
                    params.extend([user_profile_id, user_profile_id])
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += " ORDER BY c.timestamp DESC"
                
                cursor = conn.execute(query, params)
                contradictions = []
                
                for row in cursor.fetchall():
                    contradictions.append({
                        "id": row[0],
                        "fact_a_id": row[1],
                        "fact_b_id": row[2],
                        "fact_a_text": row[3],
                        "fact_b_text": row[4],
                        "confidence": row[5],
                        "resolved": bool(row[6]),
                        "timestamp": row[7]
                    })
                
                return contradictions
                
        except Exception as e:
            logging.error(f"Error getting contradictions: {e}")
            return [] 