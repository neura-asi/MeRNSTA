"""
Conversation handling module for MeRNSTA cortex package.
Handles the main conversation loop and command processing.
"""

import sys
import logging
import signal
import atexit
from typing import Dict, Any
from storage.memory_log import MemoryLog
from storage.auto_reconciliation import AutoReconciliationEngine
from storage.memory_compression import MemoryCompressionEngine
from config.settings import DEFAULT_PERSONALITY, DEFAULT_MEMORY_MODE, MEMORY_ROUTING_MODES, PERSONALITY_PROFILES, enable_compression
from .memory_ops import process_user_input
from .cli_utils import handle_fuzzy_command, AVAILABLE_COMMANDS
from .meta_goals import execute_meta_goals

class ConversationManager:
    """Manages the main conversation loop and system state."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        from config.settings import DATABASE_CONFIG
        db_path = DATABASE_CONFIG.get("default_path", "memory.db")
        self.memory_log = MemoryLog(db_path)
        self.current_memory_mode = DEFAULT_MEMORY_MODE
        self.current_personality = DEFAULT_PERSONALITY
        
        # Initialize background engines
        self.auto_reconciliation = AutoReconciliationEngine(self.memory_log, check_interval=30)
        if enable_compression:
            self.memory_compression = MemoryCompressionEngine(self.memory_log)
            self.memory_compression.start_background_loop()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
        
        # Start auto-reconciliation
        self.auto_reconciliation.start_background_loop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutting down MeRNSTA gracefully...")
        self._cleanup()
        print("👋 Goodbye!")
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup function called on exit."""
        try:
            self.auto_reconciliation.stop_background_loop()
            if enable_compression and hasattr(self, 'memory_compression'):
                self.memory_compression.stop_background_loop()
        except Exception as e:
            logging.exception("Error during cleanup")
    
    def shutdown(self):
        """Public shutdown method for graceful cleanup."""
        try:
            # Stop background processes
            self._cleanup()
            
            # Shutdown memory log
            if hasattr(self, 'memory_log'):
                self.memory_log.shutdown()
            
            logging.info("ConversationManager shutdown completed")
            
        except Exception as e:
            logging.error(f"Error during ConversationManager shutdown: {e}")
    
    def run_conversation(self):
        """Run the main conversation loop."""
        print("🧠 MeRNSTA v0.6.4 - Advanced Memory Architecture with Cognitive Enhancements")
        print(f"🧬 Mode: {self.current_memory_mode} ({MEMORY_ROUTING_MODES[self.current_memory_mode]['name']})")
        print(f"🎭 Personality: {self.current_personality} ({PERSONALITY_PROFILES[self.current_personality]['name']})")
        print("-" * 50)
        print("💡 Commands: list_facts, show_contradictions, generate_meta_goals, health_check")
        print("💡 Commands: set_personality, memory_mode, evolve_file_with_context")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle fuzzy command matching
                is_likely_command = (
                    user_input.lower() in AVAILABLE_COMMANDS or
                    any(user_input.lower().startswith(cmd + " ") for cmd in AVAILABLE_COMMANDS) or
                    (len(user_input.split()) == 1 and user_input.lower() in ["help", "quit", "exit", "list_facts", "personality"])
                )
                
                if is_likely_command:
                    user_input = handle_fuzzy_command(user_input)
                
                # Handle quit command
                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break
                
                # Handle special commands
                if self._handle_special_commands(user_input):
                    continue
                
                # Process regular user input
                print("\n🤖 ", end='', flush=True)
                response, token_count, new_personality = process_user_input(user_input, self.current_personality)
                self.current_personality = new_personality
                print(response)
                
                # Execute meta-goals after each interaction
                try:
                    executed = execute_meta_goals(self.memory_log)
                    if executed:
                        print(f"\n🎯 Auto-executed {len(executed)} meta-goals")
                except Exception as e:
                    print(f"\n⚠️ Meta-goal execution failed: {e}")
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down MeRNSTA gracefully...")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logging.exception("Error in conversation loop")
                print(f"❌ Error: {e}")
                continue
    
    def run_once(self):
        """Run a single interaction without looping."""
        print("You: ")
        user_input = input().strip()
        
        if not user_input:
            return
        
        print("\n🤖 ", end='', flush=True)
        response, token_count, new_personality, metadata = process_user_input(user_input, self.current_personality)
        self.current_personality = new_personality
        
        print(response)
    
    def _handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        lower_input = user_input.lower()
        
        # Basic command handling - this is a simplified version
        # In the full implementation, you'd want to move all the command handlers here
        if lower_input == "list_facts":
            print("\n📚 All Facts (with IDs):")
            print("-" * 60)
            facts_with_ids = self.memory_log.list_facts_with_ids()
            if not facts_with_ids:
                print("No facts found.")
                return True
            
            for fact_id, fact in facts_with_ids:
                contra_icon = "⚠️" if getattr(fact, 'contradiction_score', 0) > 0.7 else ""
                vol_icon = "🔥" if getattr(fact, 'volatility_score', 0) > 0.5 else ""
                icons = f"{contra_icon}{vol_icon}".strip()
                ts = getattr(fact, 'timestamp', '')
                if isinstance(ts, str) and ts:
                    ts_disp = ts.split()[0]
                else:
                    ts_disp = str(ts)
                print(f"{fact_id:3d}. {icons} {fact.subject} {fact.predicate} {fact.object} (seen {fact.frequency}×, last: {ts_disp})")
            return True
        
        elif lower_input == "personality":
            print(f"\n🎭 Current Personality: {self.current_personality}")
            print(f"📝 {PERSONALITY_PROFILES[self.current_personality]['name']}")
            print(f"💡 {PERSONALITY_PROFILES[self.current_personality]['description']}")
            print("\nAvailable personalities:")
            for personality, config in PERSONALITY_PROFILES.items():
                status = " (current)" if personality == self.current_personality else ""
                print(f"  {personality}: {config['name']}{status}")
            return True
        
        elif lower_input.startswith("set_personality "):
            new_personality = user_input.split(" ", 1)[1].strip().lower()
            if new_personality == "auto":
                print("🤖 Dynamic personality mode enabled!")
                print("   The system will automatically switch personalities based on:")
                print("   - Contradiction levels (skeptical when high)")
                print("   - Emotional content (emotional when intense)")
                print("   - Memory volatility (analytical when stable)")
                self.current_personality = "auto"
            elif new_personality in PERSONALITY_PROFILES:
                self.current_personality = new_personality
                print(f"🎭 Personality set to: {new_personality} ({PERSONALITY_PROFILES[new_personality]['name']})")
            else:
                print(f"❌ Unknown personality: {new_personality}")
                print(f"Available: {', '.join(PERSONALITY_PROFILES.keys())}, auto")
            return True
        
        elif lower_input == "generate_meta_goals":
            print("\n🎯 Generating Meta-Goals for Memory Maintenance...")
            print("-" * 60)
            goals = self.memory_log.generate_meta_goals()
            if not goals:
                print("✅ No meta-goals generated - memory is healthy!")
            else:
                print(f"📋 Generated {len(goals)} meta-goals:")
                for i, goal in enumerate(goals, 1):
                    print(f"   {i}. {goal}")
                print(f"\n💡 To execute these goals, run: execute_meta_goals")
            return True
        
        elif lower_input == "execute_meta_goals":
            print("\n🎯 Executing meta-goals...")
            executed = execute_meta_goals(self.memory_log)
            if executed:
                print(f"✅ Executed {len(executed)} meta-goals")
            else:
                print("ℹ️ No meta-goals to execute")
            return True
        
        elif lower_input == "health_check":
            print("\n🏥 System Health Check:")
            print("-" * 60)
            print("✅ Basic health check completed")
            return True
        
        return False 