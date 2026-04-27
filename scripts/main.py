#!/usr/bin/env python3
"""
MeRNSTA Unified Main Entry Point
Universal launcher for all MeRNSTA system modes and configurations.
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print the MeRNSTA banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🧠 MeRNSTA v1.0.0                        ║
║          Memory-Ranked Neuro-Symbolic Transformer           ║
║                                                              ║
║  🚀 Autonomous Cognitive AGI System                         ║
║  🤖 23 Specialized Agents • Multi-Modal Memory             ║
║  ⚡ Web Chat • CLI • API • Enterprise Features             ║
╚══════════════════════════════════════════════════════════════╝
""")

def setup_logging(level=logging.INFO):
    """Setup unified logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mernsta.log')
        ]
    )

async def run_web_mode(args):
    """Run web interface mode."""
    print("🌐 Starting MeRNSTA Web Interface...")
    
    try:
        from web.main import app
        import uvicorn
        
        host = args.host or '0.0.0.0'
        port = args.port or 8000
        
        print(f"💬 Web Chat: http://{host}:{port}/chat")
        print(f"📊 Health: http://{host}:{port}/health")
        print(f"🤖 Agents: http://{host}:{port}/agents/status")
        
        uvicorn.run(
            "web.main:app",
            host=host,
            port=port,
            reload=args.reload,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

async def run_cli_mode(args):
    """Run CLI shell mode."""
    print("💻 Starting MeRNSTA CLI Shell...")
    
    try:
        from cli.mernsta_shell import MeRNSTAShell
        
        shell = MeRNSTAShell()
        await shell.run()
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")
        sys.exit(1)

async def run_api_mode(args):
    """Run API server only mode."""
    print("🔌 Starting MeRNSTA API Server...")
    
    try:
        from api.system_bridge import SystemBridgeAPI
        
        api = SystemBridgeAPI()
        host = args.host or '0.0.0.0'
        port = args.port or 8001
        
        print(f"🔌 API Server: http://{host}:{port}")
        print(f"📖 API Docs: http://{host}:{port}/docs")
        
        api.run(host=host, port=port, reload=args.reload)
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        sys.exit(1)

async def run_integration_mode(args):
    """Run full system integration mode."""
    print("🔧 Starting MeRNSTA Integration Mode...")
    
    try:
        from system.integration_runner import MeRNSTAIntegrationRunner
        
        mode = getattr(args, 'integration_mode', None) or 'daemon'
        runner = MeRNSTAIntegrationRunner(mode=mode)
        
        print(f"🚀 Integration mode: {mode}")
        print(f"🧠 Cognitive agents: Active")
        print(f"🔄 Background tasks: Running")
        
        await runner.start()
    except Exception as e:
        print(f"❌ Error in integration mode: {e}")
        sys.exit(1)

async def run_enterprise_mode(args):
    """Run enterprise production mode."""
    print("🏢 Starting MeRNSTA Enterprise Mode...")
    
    try:
        import subprocess
        import time
        
        # Use the existing enterprise starter
        print("🔄 Starting enterprise services...")
        result = subprocess.run([sys.executable, "start_enterprise.py"], 
                              capture_output=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error starting enterprise mode: {e}")
        sys.exit(1)

async def run_unified_mode(args):
    """Run unified full AGI mode - all components at once."""
    print("🚀 Starting MeRNSTA Unified Full AGI Mode...")
    
    try:
        from system.unified_runner import MeRNSTAUnifiedRunner
        
        runner = MeRNSTAUnifiedRunner(
            web_port=args.web_port or 8000,
            api_port=args.api_port or 8001,
            enable_web=not args.no_web,
            enable_api=not args.no_api,
            enable_background=not args.no_background,
            enable_agents=not args.no_agents,
            enable_enterprise=args.enterprise,
            debug=args.debug
        )
        
        print("🧠 Full AGI System Starting:")
        print(f"  💬 Web Chat UI: {'Enabled' if runner.enable_web else 'Disabled'}")
        print(f"  🔌 API Server: {'Enabled' if runner.enable_api else 'Disabled'}")
        print(f"  🤖 Cognitive Agents: {'Enabled' if runner.enable_agents else 'Disabled'}")
        print(f"  🔄 Background Tasks: {'Enabled' if runner.enable_background else 'Disabled'}")
        print(f"  🏢 Enterprise Features: {'Enabled' if runner.enable_enterprise else 'Disabled'}")
        
        await runner.start()
    except Exception as e:
        print(f"❌ Error starting unified mode: {e}")
        sys.exit(1)

def run_interactive_mode(args):
    """Run simple interactive mode."""
    print("🎮 Starting MeRNSTA Interactive Mode...")
    
    try:
        from cli.mernsta_shell import MeRNSTAShell
        shell = MeRNSTAShell()
        asyncio.run(shell.run())
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")
        sys.exit(1)

def create_parser():
    """Create the unified argument parser."""
    parser = argparse.ArgumentParser(
        description='MeRNSTA - Unified Autonomous Cognitive AGI System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run                           # Full AGI mode (recommended)
  python main.py web                           # Start web interface only
  python main.py cli                           # Start CLI shell
  python main.py api                           # Start API server only
  python main.py integration --mode daemon    # Full system integration
  python main.py enterprise                   # Production enterprise mode
  python main.py interactive                  # Simple interactive mode
  
  python main.py run --web-port 8080          # Full AGI mode with custom web port
  python main.py run --no-web                 # Full AGI mode without web interface
  python main.py run --enterprise             # Full AGI mode with enterprise features
  python main.py web --port 8080              # Web on custom port
  python main.py api --host 127.0.0.1         # API on localhost only
  python main.py integration --mode headless  # Headless integration mode
        """
    )
    
    # Main mode selector
    parser.add_argument(
        'mode',
        choices=['web', 'cli', 'api', 'integration', 'enterprise', 'interactive', 'run'],
        help='MeRNSTA operation mode'
    )
    
    # Common options
    parser.add_argument('--host', help='Host to bind to (for web/api modes)')
    parser.add_argument('--port', type=int, help='Port to bind to (for web/api modes)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    # Integration mode specific options
    parser.add_argument(
        '--integration-mode',
        choices=['daemon', 'interactive', 'headless', 'bridge_only'],
        help='Integration runner mode'
    )
    
    # Unified mode specific options
    parser.add_argument('--web-port', type=int, help='Web server port (default: 8000)')
    parser.add_argument('--api-port', type=int, help='API server port (default: 8001)')
    parser.add_argument('--no-web', action='store_true', help='Disable web chat interface')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('--no-background', action='store_true', help='Disable background tasks')
    parser.add_argument('--no-agents', action='store_true', help='Disable cognitive agents')
    parser.add_argument('--enterprise', action='store_true', help='Enable enterprise features')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging(logging.DEBUG)
    elif args.quiet:
        setup_logging(logging.WARNING)
    else:
        setup_logging(logging.INFO)
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Ollama pre-flight check (except for interactive mode)
    if args.mode != 'interactive':
        try:
            from utils.ollama_checker import validate_ollama_setup, ensure_ollama_ready
            
            print("🔍 Checking Ollama setup...")
            is_valid, message = validate_ollama_setup()
            
            if not is_valid:
                print(f"⚠️ Ollama setup issue: {message}")
                print("🚀 Attempting to start Ollama automatically...")
                
                if ensure_ollama_ready():
                    print("✅ Ollama is now ready!")
                else:
                    # Do NOT exit; continue with graceful LLM fallback
                    print("❌ Failed to start Ollama automatically — continuing with LLM fallback only")
                    print("ℹ️ You can manually start it: cd external/ollama && ./ollama serve")
            else:
                print("✅ Ollama setup is valid")
                
        except ImportError:
            print("⚠️ Ollama checker not available, skipping pre-flight check")
        except Exception as e:
            print(f"⚠️ Ollama check failed: {e} — continuing with LLM fallback if needed")
    
    # Route to appropriate mode
    try:
        if args.mode == 'web':
            await run_web_mode(args)
        elif args.mode == 'cli':
            await run_cli_mode(args)
        elif args.mode == 'api':
            await run_api_mode(args)
        elif args.mode == 'integration':
            await run_integration_mode(args)
        elif args.mode == 'enterprise':
            await run_enterprise_mode(args)
        elif args.mode == 'interactive':
            run_interactive_mode(args)
        elif args.mode == 'run':
            await run_unified_mode(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())