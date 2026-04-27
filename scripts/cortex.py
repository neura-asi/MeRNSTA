#!/usr/bin/env python3
"""
MeRNSTA - Memory-Ranked Neuro-Symbolic Transformer Architecture
Main entry point for the memory-augmented language model.
Refactored to use the cortex package modules.
"""

import sys
import yaml
import logging
import signal
import atexit
from cortex import CortexEngine, ContradictionDetector, EntropyCalculator, PPOTuner
from cortex.conversation import ConversationManager
from config.settings import DEFAULT_VALUES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mernsta.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create background logger
background_logger = logging.getLogger('background')
background_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('background.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
background_logger.addHandler(file_handler)
background_logger.propagate = False

# Global variables for cleanup
conversation_manager = None

def cleanup_resources():
    """Cleanup function for graceful shutdown"""
    global conversation_manager
    
    try:
        # Shutdown conversation manager
        if conversation_manager:
            conversation_manager.shutdown()
        
        # Cleanup database connection pool
        from storage.db_utils import cleanup_database_pool
        cleanup_database_pool()
        
        background_logger.info("Application cleanup completed")
        
    except Exception as e:
        background_logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    cleanup_resources()
    print("👋 Goodbye!")
    sys.exit(0)

def main():
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher required")
        sys.exit(1)
    """Main entry point for MeRNSTA."""
    global conversation_manager
    
    # Register signal handlers and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    try:
        # Load configuration
        cfg = yaml.safe_load(open("configs/config.yaml"))
        BETA = cfg.get("beta", 5)
        GAMMA = cfg.get("gamma", 0.15)
        MODEL = cfg.get("model", "mistral")
        
        # Initialize components
        engine = CortexEngine("configs/config.yaml")
        detector = ContradictionDetector(gamma=GAMMA)
        detector.set_rules(cfg.get("facts", []))
        entropy_calc = EntropyCalculator(temperature=DEFAULT_VALUES["volatility_default"])
        ppo_tuner = PPOTuner("configs/config.yaml")
        
        # Initialize conversation manager
        conversation_manager = ConversationManager("configs/config.yaml")
        
        # Run based on command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "--repl":
            conversation_manager.run_conversation()
        else:
            conversation_manager.run_once()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        cleanup_resources()
    except Exception as e:
        logging.exception("Fatal error in main loop")
        print(f"❌ Fatal error: {e}")
        cleanup_resources()
        sys.exit(1)

if __name__ == "__main__":
    main()


