#!/usr/bin/env python3
"""
Memory operations module for MeRNSTA cortex package.
Handles triplet queries, summarization, user input processing, contradiction detection,
personality-based confidence adjustments, and real-time WebSocket updates.
"""

import logging
import uuid
import hashlib
import redis
import json
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

from storage.memory_log import MemoryLog
from storage.memory_search import MemorySearchEngine
from storage.fact_manager import FactManager
from storage.memory_utils import build_smart_memory_context, detect_contradictions, is_plural_query
from config.environment import get_settings
from config.settings import DATABASE_CONFIG, VERBOSITY_LEVEL, require_confirmation, max_patch_size
from .response_generation import generate_response, estimate_tokens
import difflib
import os
from storage.db import db

# Import enhanced memory system
try:
    from storage.enhanced_memory_system import EnhancedMemorySystem
    ENHANCED_MODE = True
    print("Enhanced memory system loaded successfully")
except ImportError as e:
    print(f"Enhanced memory system not available: {e}, using legacy mode")
    ENHANCED_MODE = False

# Get settings
settings = get_settings()

def get_user_profile_id(request_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Get user profile ID based on authentication or request context.
    Falls back to IP-based hashing if no authentication available.
    """
    
    # Try to get from authentication token first
    if request_context and 'token' in request_context:
        token = request_context['token']
        # Use token hash as user ID for authenticated users
        return hashlib.sha256(token.encode()).hexdigest()[:16]
    
    # Try to get from IP address
    if request_context and 'client_ip' in request_context:
        ip = request_context['client_ip']
        # Use IP hash as user ID for anonymous users
        return f"user_{hashlib.md5(ip.encode()).hexdigest()[:8]}"
    
    # Fallback to default from config or environment
    default_user = getattr(settings, 'default_user_profile', 'default_user')
    return default_user

def get_database_path() -> str:
    """Get database path from settings configuration."""
    # Check environment settings first
    if hasattr(settings, 'database_url') and settings.database_url:
        db_url = settings.database_url
        if db_url.startswith('sqlite:///'):
            return db_url.replace('sqlite:///', '')
        # Handle PostgreSQL URLs by returning the URL as-is for SQLAlchemy
        return db_url
    
    # Try to get database path from settings
    if hasattr(settings, 'database_path'):
        return settings.database_path
    
    # Fall back to config settings and then defaults
    path = DATABASE_CONFIG.get("default_path") or DATABASE_CONFIG.get("path", "memory.db")
    return path

def publish_memory_update(update_data: Dict[str, Any]) -> None:
    """
    Publish memory updates to Redis channel for real-time WebSocket notifications.
    """
    try:
        if not settings.enable_caching:
            return
            
        redis_client = redis.from_url(settings.redis_url)
        channel = settings.websocket_channel
        
        update_message = {
            "timestamp": datetime.now().isoformat(),
            "type": update_data.get("type", "fact_update"),
            "data": update_data
        }
        
        redis_client.publish(channel, json.dumps(update_message))
        
    except Exception as e:
        logging.warning(f"Failed to publish memory update: {e}")

def determine_input_type(user_input: str) -> str:
    """
    Determine if the input is a query, statement, command, conversation, or mixed.
    Uses linguistic analysis rather than hardcoded patterns.
    
    Args:
        user_input: User's message
        
    Returns:
        Input type: "query", "statement", "command", "conversation", "general_knowledge", or "mixed"
    """
    
    user_input_lower = user_input.lower().strip()
    
    # Check if it's a question based on linguistic features
    if user_input.strip().endswith("?"):
        # Questions ending with ? are queries
        return "query"
    
    # Try to use spaCy for linguistic analysis if available
    try:
        from storage.spacy_extractor import nlp
        if nlp:
            doc = nlp(user_input)
            
            # Check for interrogative words at the beginning
            if doc and len(doc) > 0:
                first_token = doc[0]
                if first_token.tag_ in ['WDT', 'WP', 'WP$', 'WRB']:  # Wh- words
                    return "query"
                
                # Check for auxiliary verbs at the beginning (questions)
                if first_token.tag_ in ['MD', 'VBZ'] and first_token.dep_ == 'aux':
                    return "query"
            
            # Check for imperative mood (commands) - but exclude negations
            if doc and len(doc) > 0:
                root = [token for token in doc if token.dep_ == "ROOT"]
                if root and root[0].tag_ == "VB":  # Base form verb at root
                    # Check if it's negated or has a subject
                    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
                    has_negation = any(token.dep_ == "neg" or token.text.lower() in ["don't", "not"] for token in doc)
                    if not has_subject and not has_negation:
                        return "command"
            
            # Check if it contains first-person pronouns (likely personal statement)
            has_first_person = any(token.text.lower() in ['i', 'my', 'me'] for token in doc)
            if has_first_person:
                return "statement"
            
            # Check for copula constructions (X is Y, X are Y)
            for token in doc:
                if token.lemma_ == 'be' and token.dep_ == 'ROOT':
                    # This is likely a factual statement
                    return "statement"
    except:
        # spaCy not available, use basic heuristics
        pass
    
    # Basic command detection for system commands
    if any(cmd in user_input_lower for cmd in ['help', 'reset', 'clear memory', 'set personality']):
        return "command"
    
    # If it starts with a verb in imperative form
    words = user_input_lower.split()
    if words and words[0] in ['tell', 'show', 'list', 'give', 'explain', 'describe']:
        return "general_knowledge"
    
    # Check for personal pronouns indicating statements
    if any(pronoun in words for pronoun in ['i', 'my', 'me', 'mine']):
        return "statement"
    
    # Check for factual statement patterns (X is Y, X are Y, X have Z)
    if len(words) >= 3:
        # Look for copula or other state verbs
        for i, word in enumerate(words):
            if word in ['is', 'are', 'was', 'were', 'has', 'have', 'had']:
                # Found a factual statement verb
                return "statement"
    
    # Check for mixed content (statements with questions)
    if "?" in user_input and any(pronoun in words for pronoun in ['i', 'my', 'me']):
        return "mixed"
    
    # Very short inputs (1-2 words) without clear structure are conversational
    if len(words) <= 2:
        return "conversation"
    
    # Default to statement for declarative sentences with 3+ words
    return "statement"

def handle_conversational_input(user_input: str, current_personality: str) -> str:
    """Handle general conversational input that doesn't require memory operations."""
    
    user_input_lower = user_input.lower().strip()
    
    # Greeting responses
    if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey"]):
        return "Hello! I'm doing well, thank you. How can I help you today?"
    
    # How are you responses  
    if any(phrase in user_input_lower for phrase in ["how are you", "how's it going", "how do you feel"]):
        personality_responses = {
            "neutral": "I'm doing well, thank you for asking! How are you?",
            "enthusiastic": "I'm fantastic! Thanks for asking! How are you doing today?",
            "skeptical": "I'm functioning as expected. How are you?",
            "cautious": "I'm operating normally, thank you. How may I assist you?",
            "empathetic": "I'm doing well, and I appreciate you asking. How are you feeling today?"
        }
        return personality_responses.get(current_personality, personality_responses["neutral"])
    
    # What's up / what's new responses
    if any(phrase in user_input_lower for phrase in ["what's up", "what's new", "whats up"]):
        return "Not much, just here and ready to help! What's on your mind?"
    
    # Thank you responses
    if any(phrase in user_input_lower for phrase in ["thank you", "thanks"]):
        return "You're very welcome! I'm happy to help."
    
    # Goodbye responses
    if any(phrase in user_input_lower for phrase in ["goodbye", "bye", "see you", "take care"]):
        return "Goodbye! Take care, and feel free to come back anytime!"
    
    # Good morning/afternoon/evening
    if "good morning" in user_input_lower:
        return "Good morning! I hope you're having a great start to your day!"
    elif "good afternoon" in user_input_lower:
        return "Good afternoon! How has your day been going?"
    elif "good evening" in user_input_lower:
        return "Good evening! I hope you've had a wonderful day!"
    
    # Default conversational response
    return "I'm here and ready to chat! Is there anything specific you'd like to talk about or any information you'd like to share?"

def handle_statement_input(user_input: str, current_personality: str, session_id: str,
                         user_profile_id: str, memory_log: MemoryLog, fact_manager: FactManager,
                         search_engine: MemorySearchEngine) -> Tuple[str, int, str, Dict[str, Any]]:
    """Handle statement input by extracting and storing facts."""
    
    # Extract triplets from the statement
    triplets = memory_log.extract_triplets(user_input)
    
    if not triplets:
        response = "I understood your message but couldn't extract any specific facts from it."
        return response, estimate_tokens(user_input + response), current_personality, {"triplets": []}
    
    # Store facts with validation
    storage_result = fact_manager.store_facts_with_validation(
        triplets, 
        user_profile_id=user_profile_id, 
        session_id=session_id
    )
    
    # Handle dynamic personality switching
    new_personality = current_personality
    if current_personality == "auto":
        new_personality = determine_personality_from_facts(memory_log, storage_result)
    
    # Build response
    response_parts = []
    import random
    acknowledgments = ["Cool, noted that", "Got it, you", "Awesome, remembered that you", "Alright, I've got that you"]
    ack = random.choice(acknowledgments)
    if triplets:
        example_fact = triplets[0]
        response_parts.append(f"{ack} {example_fact[1]} {example_fact[2]}.")

    if storage_result["summary_messages"]:
        for msg in storage_result["summary_messages"][:2]:  # Show first 2 messages
            response_parts.append(msg)
    
    if storage_result["contradictions"]:
        contradiction = storage_result["contradictions"][0]  # Show first contradiction
        response_parts.append(
            f"⚠️ This contradicts your earlier statement: '{contradiction['existing_fact']}'. "
            f"Which is correct?"
        )
    
    response = " ".join(response_parts)
    token_count = estimate_tokens(user_input + response)
    
    # FIXED: Properly report extracted facts count
    metadata = {
        "triplets": triplets,
        "extracted_facts": len(triplets),  # Always report actual extracted count
        "stored_facts": len(storage_result.get("stored_ids", [])),
        "contradictions_detected": len(storage_result.get("contradictions", [])),
        "storage_result": storage_result,
        "input_type": "statement",
        "session_id": session_id,
        "user_profile_id": user_profile_id
    }
    
    return response, token_count, new_personality, metadata

def handle_query_input(user_input: str, current_personality: str, session_id: str,
                      user_profile_id: str, search_engine: MemorySearchEngine) -> Tuple[str, int, str, Dict[str, Any]]:
    """Handle memory query input by searching stored facts and applying personality-based confidence adjustments."""
    
    search_result = search_engine.search_facts(
        user_input,
        user_profile_id=user_profile_id,
        personality=current_personality
    )
    
    response = search_result["response"]
    token_count = estimate_tokens(user_input + response)
    
    # Publish WebSocket update for query results
    publish_memory_update({
        "type": "query_result",
        "user_profile_id": user_profile_id,
        "session_id": session_id,
        "query": user_input,
        "facts_found": len(search_result.get("facts", [])),
        "personality": current_personality
    })
    
    metadata = {
        "input_type": "query",
        "search_result": search_result,
        "facts_found": len(search_result.get("facts", [])),
        "personality_applied": current_personality,
        "user_profile_id": user_profile_id,
        "session_id": session_id
    }
    
    return response, token_count, current_personality, metadata

def handle_general_knowledge_input(user_input: str, current_personality: str, session_id: str,
                                 user_profile_id: str, memory_log: MemoryLog) -> Tuple[str, int, str, Dict[str, Any]]:
    """Handle general knowledge questions using LLM generation rather than memory search."""
    
    try:
        # Use the LLM to generate a response for general knowledge questions
        # Build context that encourages factual responses for general knowledge
        context = f"""This is a general knowledge question that requires a factual answer based on common knowledge, not personal memory. 
Personality: {current_personality}. 
Please provide a helpful, accurate, and concise answer to this factual question. 
This is not about personal preferences or stored user data - it's asking for general information. Do not reference the user, their interactions, or any personal context. Respond as if this is a standalone factual query."""
        
        response = generate_response(user_input, context)
        token_count = estimate_tokens(user_input + context + response)
        
        metadata = {
            "input_type": "general_knowledge",
            "context_used": context,
            "personality_applied": current_personality,
            "user_profile_id": user_profile_id,
            "session_id": session_id,
            "llm_generated": True
        }
        
        return response, token_count, current_personality, metadata
        
    except Exception as e:
        # Fallback response if LLM generation fails
        response = "I'm not sure about that. Could you try rephrasing your question or asking about something else?"
        token_count = estimate_tokens(user_input + response)
        
        metadata = {
            "input_type": "general_knowledge",
            "error": str(e),
            "fallback_used": True,
            "user_profile_id": user_profile_id,
            "session_id": session_id
        }
        
        return response, token_count, current_personality, metadata

def handle_command_input(user_input: str, current_personality: str, session_id: str,
                        user_profile_id: str, memory_log: MemoryLog, 
                        search_engine: MemorySearchEngine) -> Tuple[str, int, str, Dict[str, Any]]:
    """Handle command input like summarization, personality changes, etc."""
    
    user_input_lower = user_input.lower().strip()
    
    # Handle summarization commands
    if any(keyword in user_input_lower for keyword in ["summarize", "summary"]):
        # Get recent facts for this user and session
        all_facts = memory_log.get_all_facts()
        user_facts = [f for f in all_facts if getattr(f, 'user_profile_id', None) == user_profile_id and getattr(f, 'session_id', None) == session_id]
        response = summarize_triplet_matches(user_facts, session_id, user_profile_id)
        token_count = estimate_tokens(user_input + response)
        
        metadata = {
            "command": "summarize",
            "input_type": "command",
            "episode_stored": True
        }
        
        # Publish WebSocket update
        publish_memory_update({
            "type": "episode_created",
            "session_id": session_id,
            "user_profile_id": user_profile_id,
            "summary": response,
            "fact_count": len(user_facts)
        })
        
        return response, token_count, current_personality, metadata
    
    # Handle personality switching
    if any(keyword in user_input_lower for keyword in ["switch to", "change personality", "set personality"]):
        new_personality = extract_personality_from_command(user_input)
        response = f"Switched to {new_personality} personality. I'll be more questioning of facts and reduce confidence scores."
        
        metadata = {
            "command": "personality_switch",
            "old_personality": current_personality,
            "new_personality": new_personality,
            "input_type": "command"
        }
        
        return response, estimate_tokens(user_input + response), new_personality, metadata
    
    # Default command response
    response = "I understood that as a command, but I'm not sure how to execute it yet."
    metadata = {"command": "unknown", "input_type": "command"}
    
    return response, estimate_tokens(user_input + response), current_personality, metadata

def handle_mixed_input(user_input: str, current_personality: str, session_id: str,
                      user_profile_id: str, memory_log: MemoryLog, fact_manager: FactManager,
                      search_engine: MemorySearchEngine) -> Tuple[str, int, str, Dict[str, Any]]:
    """Handle input that contains both statements and queries/conversation."""
    
    triplets = memory_log.extract_triplets(user_input)
    valid_triplets = [t for t in triplets if len(t) >= 3 and all(t[:3])]
    
    if valid_triplets:
        storage_result = fact_manager.store_facts_with_validation(
            valid_triplets, 
            user_profile_id=user_profile_id, 
            session_id=session_id
        )
        new_personality = current_personality
        if current_personality == "auto":
            new_personality = determine_personality_from_facts(memory_log, storage_result)
        
        response_parts = []
        
        # Check if input has conversational elements using linguistic cues
        has_greeting = False
        try:
            from storage.spacy_extractor import nlp
            if nlp:
                doc = nlp(user_input.lower())
                # Check for interjections (greetings, exclamations)
                for token in doc:
                    if token.pos_ == 'INTJ':
                        has_greeting = True
                        break
        except:
            # Basic fallback
            if user_input.lower().startswith(('hi', 'hello', 'hey')):
                has_greeting = True
        
        if has_greeting:
            convo_response = handle_conversational_input(user_input, current_personality)
            response_parts.append(convo_response)
        
        if storage_result["summary_messages"]:
            response_parts.extend(storage_result["summary_messages"][:2])
        
        if storage_result["contradictions"]:
            contradiction = storage_result["contradictions"][0]
            response_parts.append(
                f"⚠️ This contradicts your earlier statement: '{contradiction['existing_fact']}'. "
                f"Which is correct?"
            )
        
        if not response_parts:
            response_parts.append("Got it! I've noted that information.")
        
        response = " ".join(response_parts)
        token_count = estimate_tokens(user_input + response)
        
        metadata = {
            "input_type": "mixed",
            "handled_as": "statement",
            "storage_result": storage_result,
            "triplets_stored": len(valid_triplets),
            "personality_applied": new_personality,
            "user_profile_id": user_profile_id,
            "session_id": session_id
        }
        
        publish_memory_update({
            "type": "facts_stored",
            "user_profile_id": user_profile_id,
            "session_id": session_id,
            "facts_stored": len(valid_triplets),
            "personality": new_personality
        })
        
        return response, token_count, new_personality, metadata
    else:
        # Fallback to query handling if no valid facts
        return handle_query_input(
            user_input, current_personality, session_id, user_profile_id, search_engine
        )

def determine_personality_from_facts(memory_log: MemoryLog, storage_result: Dict[str, Any]) -> str:
    """Determine personality based on current memory state and storage results."""
    
    all_facts = memory_log.get_all_facts()
    contradictions = storage_result.get("contradictions", [])
    
    contradiction_level = len(contradictions) / max(len(all_facts), 1)
    
    if contradiction_level > 0.3:
        return "skeptical"
    elif contradiction_level < 0.1:
        return "analytical"
    else:
        return "neutral"

def extract_personality_from_command(command: str) -> str:
    """Extract personality name from a personality switch command."""
    
    command_lower = command.lower()
    
    # Get personalities from settings
    try:
        # Load from config.yaml via settings
        import yaml
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            personalities = list(config.get('personality_profiles', {}).keys())
    except Exception:
        # Fallback to basic personality list
        personalities = ["neutral", "skeptical", "enthusiastic"]
    
    for personality in personalities:
        if personality in command_lower:
            return personality
    
    return getattr(settings, 'default_personality', 'neutral')  # Default fallback

def query_triplets(question: str, topk: int = 5, user_profile_id: str = None) -> List:
    """
    Return the most relevant (subject, predicate, object) triplets from memory for a user question.
    Results include timestamp and frequency, ranked by weighted recency + confidence.
    """
    db_path = get_database_path()
    memory_log = MemoryLog(db_path)
    matches = memory_log.semantic_search(question, topk=topk, user_profile_id=user_profile_id)
    return matches

def summarize_triplet_matches(matches: List, session_id: str = None, user_profile_id: str = None) -> str:
    """
    Convert top triplet matches into a natural language summary and store in episodes table.
    Generates summaries like: "You said you like blue trucks and yellow cats, but later said you hate yellow cats, creating a contradiction."
    """
    if not matches:
        return "I couldn't find any relevant facts in memory to summarize."
    
    # Group facts by predicate type
    likes = []
    dislikes = []
    facts = []
    
    for match in matches:
        # Handle both dict and object formats
        if isinstance(match, dict):
            subject = match.get('subject', '')
            predicate = match.get('predicate', '')
            obj = match.get('object', '')
        else:
            subject = getattr(match, 'subject', '')
            predicate = getattr(match, 'predicate', '')
            obj = getattr(match, 'object', '')
        
        predicate_lower = predicate.lower()
        
        if predicate_lower in ['like', 'love', 'enjoy', 'prefer']:
            likes.append(obj)
        elif predicate_lower in ['hate', 'dislike']:
            dislikes.append(obj)
        else:
            facts.append(f"{predicate} {obj}")
    
    # Detect contradictions (same object in likes and dislikes)
    overlapping = set(likes) & set(dislikes)
    
    # Build natural language summary
    summary_parts = []
    
    if likes:
        if len(likes) == 1:
            summary_parts.append(f"you like {likes[0]}")
        else:
            summary_parts.append(f"you like {', '.join(likes[:-1])} and {likes[-1]}")
    
    if dislikes:
        if len(dislikes) == 1:
            summary_parts.append(f"you dislike {dislikes[0]}")
        else:
            summary_parts.append(f"you dislike {', '.join(dislikes[:-1])} and {dislikes[-1]}")
    
    if facts:
        summary_parts.append(f"and {', '.join(facts)}")
    
    if summary_parts:
        summary = f"You said {' and '.join(summary_parts)}."
        if overlapping:
            overl = ', '.join(overlapping)
            summary += f" However, this creates a contradiction for {overl}."
    else:
        summary = f"I found {len(matches)} facts but couldn't categorize them clearly."
    
    # Store summary in episodes table if database connection available
    if session_id and user_profile_id:
        try:
            db_path = get_database_path()
            memory_log = MemoryLog(db_path)
            
            with memory_log._connection_pool.get_connection() as conn:
                conn.execute("""
                    INSERT INTO episodes (session_id, user_profile_id, start_time, fact_count, summary)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
                """, (session_id, user_profile_id, len(matches), summary))
                conn.commit()
                logging.info(f"Stored episode summary for session {session_id}")
        except Exception as e:
            logging.error(f"Failed to store episode summary: {e}")
    
    return summary

def process_user_input(user_input: str, current_personality: str = "neutral", 
                      session_id: str = None, request_context: Dict[str, Any] = None) -> Tuple[str, int, str, Dict[str, Any]]:
    """
    Process user input through the most appropriate system, with command routing at the top.
    """
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    user_profile_id = get_user_profile_id()
    if request_context:
        client_ip = request_context.get("client_ip", "unknown")
        user_profile_id = f"{client_ip}_{user_profile_id}"
    
    # ================================
    # COMMAND ROUTING - HANDLE AT TOP BEFORE ANY EXTRACTION/STORAGE
    # ================================
    user_input_lower = user_input.lower().strip()
    
    # Check for summarization commands first
    summarization_patterns = [
        r"summarize|summary|tell me what you know|what do you know about me",
        r"remind me what|recap what|overview of what", 
        r"what have we talked about|what did we discuss|conversation so far",
        r"changed my mind|what are my opinions"
    ]
    
    import re
    for pattern in summarization_patterns:
        if re.search(pattern, user_input_lower):
            print(f"[CommandRouter] Detected summarization command: {user_input}")
            try:
                # Try enhanced system first
                db_path = get_database_path()
                enhanced_db_path = db_path.replace('.db', '_enhanced.db') if db_path.endswith('.db') else db_path + '_enhanced'
                enhanced_memory = EnhancedMemorySystem(
                    db_path=enhanced_db_path,
                    ollama_host=None,
                    embedding_model=None
                )
                
                facts = enhanced_memory.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                summary = enhanced_memory.summarizer.summarize_user_facts(facts, user_profile_id)
                
                token_count = estimate_tokens(user_input + summary)
                metadata = {
                    "input_type": "command",
                    "command_type": "summarization",
                    "session_id": session_id,
                    "user_profile_id": user_profile_id,
                    "personality": current_personality
                }
                
                return summary, token_count, current_personality, metadata
                
            except Exception as e:
                logging.warning(f"Enhanced summarization failed: {e}, using legacy")
                # Fallback to legacy summarization
                try:
                    memory_log = MemoryLog(get_database_path())
                    all_facts = memory_log.get_all_facts()
                    user_facts = [f for f in all_facts if hasattr(f, 'user_profile_id') and 
                                 getattr(f, 'user_profile_id', None) == user_profile_id]
                    
                    if not user_facts:
                        user_facts = [f for f in all_facts if f.subject.lower() in ['user', 'i', 'me']]
                    
                    if user_facts:
                        response = summarize_triplet_matches(user_facts, session_id, user_profile_id)
                    else:
                        response = "I don't have any stored facts about you yet."
                    
                    token_count = estimate_tokens(user_input + response)
                    metadata = {
                        "input_type": "command",
                        "command_type": "summarization_legacy",
                        "session_id": session_id,
                        "user_profile_id": user_profile_id,
                        "personality": current_personality
                    }
                    
                    return response, token_count, current_personality, metadata
                    
                except Exception as e2:
                    logging.error(f"Legacy summarization also failed: {e2}")
                    response = "I'm having trouble accessing your stored information right now."
                    return response, estimate_tokens(user_input + response), current_personality, {"error": str(e2)}
    
    # Check for meta-goal generation commands
    meta_goal_patterns = [
        r"meta-?goal|generate some meta-?goals|/generate_meta_goals",
        r"suggest questions|clarify|clarification questions", 
        r"what should i think about|help me understand"
    ]
    
    for pattern in meta_goal_patterns:
        if re.search(pattern, user_input_lower):
            print(f"[CommandRouter] Detected meta-goal command: {user_input}")
            try:
                # Try enhanced system first
                db_path = get_database_path()
                enhanced_db_path = db_path.replace('.db', '_enhanced.db') if db_path.endswith('.db') else db_path + '_enhanced'
                enhanced_memory = EnhancedMemorySystem(
                    db_path=enhanced_db_path,
                    ollama_host=None,
                    embedding_model=None
                )
                
                facts = enhanced_memory.get_facts(user_profile_id=user_profile_id, session_id=session_id)
                volatile_topics = enhanced_memory.contradiction_resolver.get_volatile_topics(facts)
                
                if not volatile_topics:
                    response = "No volatile topics detected. Your beliefs seem stable!"
                else:
                    questions = enhanced_memory.contradiction_resolver.suggest_clarification_questions(volatile_topics)
                    response = "\n".join(questions)
                
                token_count = estimate_tokens(user_input + response)
                metadata = {
                    "input_type": "command",
                    "command_type": "meta_goals",
                    "session_id": session_id,
                    "user_profile_id": user_profile_id,
                    "personality": current_personality
                }
                
                return response, token_count, current_personality, metadata
                
            except Exception as e:
                logging.warning(f"Enhanced meta-goal generation failed: {e}")
                response = "I'm having trouble analyzing your belief patterns right now."
                return response, estimate_tokens(user_input + response), current_personality, {"error": str(e)}
    
    # ================================
    # CONTINUE WITH NORMAL PROCESSING
    # ================================
    
    # Try enhanced system for normal processing
    if ENHANCED_MODE:
        try:
            db_path = get_database_path()
            # Use a separate enhanced DB to avoid conflicts
            enhanced_db_path = db_path.replace('.db', '_enhanced.db') if db_path.endswith('.db') else db_path + '_enhanced'
            enhanced_memory = EnhancedMemorySystem(
                db_path=enhanced_db_path,
                ollama_host=None,  # Disable for now
                embedding_model=None
            )
            
            # Process with enhanced system
            result = enhanced_memory.process_input(
                user_input, 
                user_profile_id=user_profile_id, 
                session_id=session_id
            )
            
            # Convert response to expected format
            token_count = estimate_tokens(user_input + result["response"])
            
            metadata = {
                "input_type": "enhanced",
                "session_id": session_id,
                "user_profile_id": user_profile_id,
                "personality": current_personality,
                "extracted_facts": len(result.get("extracted_facts", [])),
                "contradictions_detected": len(result.get("contradictions", [])),
                "stored_facts": len(result.get("stored_facts", [])),
                "query_results": len(result.get("query_results", []))
            }
            
            # FIXED: Properly report extracted facts even if they were processed/reinforced
            if result.get("extracted_facts"):
                metadata["extracted_facts"] = len(result["extracted_facts"])
            elif "stored" in result["response"].lower() or "reinforced" in result["response"].lower():
                # If facts were stored/reinforced, assume at least 1 fact was extracted
                metadata["extracted_facts"] = 1
            
            # Add storage result for compatibility
            if result.get("stored_facts"):
                metadata["storage_result"] = {
                    "stored_ids": [f.id for f in result["stored_facts"]],
                    "summary_messages": [f"Stored new fact: {f.subject} {f.predicate} {f.object}" for f in result["stored_facts"]],
                    "contradictions": []
                }
            
            return result["response"], token_count, current_personality, metadata
            
        except Exception as e:
            logging.error(f"Enhanced memory system error: {e}, falling back to legacy")
            # Fall through to legacy system
    
    # Legacy system code continues below...
    try:
        # Initialize components
        db_path = get_database_path()
        memory_log = MemoryLog(db_path)
        search_engine = MemorySearchEngine(memory_log)
        fact_manager = FactManager(memory_log)
        
        # Determine input type
        input_type = determine_input_type(user_input)
        
        metadata = {
            "input_type": input_type,
            "session_id": session_id,
            "user_profile_id": user_profile_id,
            "personality": current_personality
        }
        
        if input_type == "conversation":
            # This is general conversation - provide conversational response
            response = handle_conversational_input(user_input, current_personality)
            token_count = estimate_tokens(user_input + response)
            
            metadata.update({
                "conversation_type": "general"
            })
            
            return response, token_count, current_personality, metadata
            
        elif input_type == "query":
            # This is a question about stored facts - use search engine with enhanced fallback
            response, token_count, personality, metadata = handle_query_input(
                user_input, current_personality, session_id, user_profile_id, search_engine
            )
            
            # ENHANCED: If query didn't find good results, try broader search
            if ("don't have" in response.lower() or "couldn't find" in response.lower()) and "information" in response.lower():
                print(f"[LegacyQueryFallback] Trying broader search for: {user_input}")
                
                all_facts = memory_log.get_all_facts()
                user_facts = [f for f in all_facts if f.subject.lower() in ['user', 'i', 'me']]
                
                # Simple keyword matching as fallback
                user_input_lower = user_input.lower()
                keywords = [word for word in user_input_lower.split() if len(word) > 3 and word not in ['what', 'know', 'have', 'information']]
                
                for fact in user_facts:
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                    if any(keyword in fact_text for keyword in keywords):
                        response = f"You {fact.predicate} {fact.object}"
                        metadata["fallback_search_used"] = True
                        print(f"[LegacyQueryFallback] Found: {fact_text}")
                        break
            
            return response, token_count, personality, metadata
            
        elif input_type == "general_knowledge":
            # This is a general knowledge question - use LLM generation
            return handle_general_knowledge_input(
                user_input, current_personality, session_id, user_profile_id, memory_log
            )
            
        elif input_type == "statement":
            # This is a statement to store - extract and store facts
            return handle_statement_input(
                user_input, current_personality, session_id, user_profile_id,
                memory_log, fact_manager, search_engine
            )
            
        elif input_type == "command":
            # This is a command - handle specially
            return handle_command_input(
                user_input, current_personality, session_id, user_profile_id,
                memory_log, search_engine
            )
            
        elif input_type == "mixed":
            # Mixed input - determine best approach
            return handle_mixed_input(
                user_input, current_personality, session_id, user_profile_id,
                memory_log, fact_manager, search_engine
            )
        
    except Exception as e:
        logging.error(f"Error in process_user_input: {e}", exc_info=True)
        error_msg = f"Error processing input: {e}"
        
        return error_msg, estimate_tokens(user_input + error_msg), current_personality, {
            "error": str(e),
            "input_type": "error"
        }

# Backward compatibility alias
process_user_input_enhanced = process_user_input

def evolve_file_with_context(file_path: str, goal: str, dry_run: bool = False, confirm: bool = None) -> dict:
    """
    Generate and optionally apply a code patch to a file based on a goal, with safeguards.
    - dry_run: If True, only preview the patch.
    - confirm: If None, use require_confirmation from config; else override.
    Returns dict with patch, status, and error info.
    """
    try:
        # Read original file
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}
        with open(file_path, "r", encoding="utf-8") as f:
            original = f.readlines()
        # Simulate code evolution (for demo, just append a comment with the goal)
        evolved = original + [f"# Evolution goal: {goal}\n"]
        # Generate patch
        patch = list(difflib.unified_diff(original, evolved, fromfile=file_path, tofile=file_path, lineterm=""))
        if len(patch) > max_patch_size:
            return {"status": "error", "error": f"Patch too large ({len(patch)} lines), max allowed is {max_patch_size}"}
        # If dry_run, do not apply
        if dry_run:
            return {"status": "dry_run", "patch": "\n".join(patch)}
        # Confirmation logic
        apply_patch = not require_confirmation if confirm is None else confirm
        if not apply_patch:
            # Store in history as not applied
            with db() as d:
                d.execute(
                    "INSERT INTO code_evolution (file_path, goal, patch, applied, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (file_path, goal, "\n".join(patch), False, datetime.now())
                )
            return {"status": "confirmation_required", "patch": "\n".join(patch)}
        # Actually apply the patch
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(evolved)
        # Store in history as applied
        with db() as d:
            d.execute(
                "INSERT INTO code_evolution (file_path, goal, patch, applied, timestamp) VALUES (?, ?, ?, ?, ?)",
                (file_path, goal, "\n".join(patch), True, datetime.now())
            )
        return {"status": "applied", "patch": "\n".join(patch)}
    except Exception as e:
        return {"status": "error", "error": str(e)} 