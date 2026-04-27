#!/usr/bin/env python3
"""
Quick start script for MeRNSTA Enterprise System.
This script helps you start all components of the enterprise deployment.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
import socket
from config.settings import api_port, port_retry_attempts
from monitoring.logger import get_logger

def print_banner():
    """Print the MeRNSTA enterprise banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    MeRNSTA Enterprise                        ║
    ║              Memory-Augmented AI System                      ║
    ║                                                              ║
    ║  🚀 Production-ready with monitoring, caching, and scaling  ║
    ║  📊 Prometheus metrics, structured logging, health checks   ║
    ║  🔄 Celery task queue, Redis caching, hot-reload config    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'celery', 'redis', 'structlog',
        'prometheus_client', 'psutil', 'pydantic', 'pydantic-settings'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_environment():
    """Check environment configuration."""
    print("🔧 Checking environment configuration...")
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  No .env file found. Creating from template...")
        create_env_file()
    
    # Check required environment variables
    required_vars = [
        'API_SECURITY_TOKEN',
        'DATABASE_URL',
        'REDIS_URL',
        'CELERY_BROKER_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Please set these in your .env file")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def create_env_file():
    """Create a basic .env file from template."""
    env_content = """# Database Configuration
DATABASE_URL=sqlite:///memory.db
MAX_CONNECTIONS=10
DATABASE_TIMEOUT=30.0

# Memory System Configuration
MAX_FACTS=1000000
COMPRESSION_THRESHOLD=0.8
MIN_CLUSTER_SIZE=3
SIMILARITY_THRESHOLD=0.7

# Background Tasks Configuration
RECONCILIATION_INTERVAL=300
COMPRESSION_INTERVAL=3600
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring Configuration
METRICS_PORT=9090
LOG_LEVEL=INFO
ENABLE_TRACING=true
PROMETHEUS_ENABLED=true

# Security Configuration
API_SECURITY_TOKEN=your-secure-token-here-change-this
RATE_LIMIT=100
RATE_LIMIT_WINDOW=60
DISABLE_RATE_LIMIT=false

# Cache Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
ENABLE_CACHING=true

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=mistral

# Feature Flags
ENABLE_COMPRESSION=true
ENABLE_AUTO_RECONCILIATION=true
ENABLE_PERSONALITY_BIASING=true
ENABLE_EMOTION_ANALYSIS=true

# Performance Configuration
BATCH_SIZE=1000
EMBEDDING_CACHE_SIZE=10000
MAX_CONCURRENT_TASKS=10

# Environment
ENVIRONMENT=development
DEBUG=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file. Please review and update the API_SECURITY_TOKEN!")

def start_redis():
    """Start Redis server."""
    print("🔴 Starting Redis...")
    try:
        # Try to connect to existing Redis
        import redis
        r = redis.from_url("redis://localhost:6379/0")
        r.ping()
        print("✅ Redis already running")
        return True
    except:
        print("⚠️  Redis not running. Please start Redis manually:")
        print("   redis-server")
        return False

def start_celery_worker():
    """Start Celery worker."""
    print("🔄 Starting Celery worker...")
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "celery", "-A", "tasks.task_queue", 
            "worker", "--loglevel=info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if process.poll() is None:
            print("✅ Celery worker started")
            return process
        else:
            print("❌ Failed to start Celery worker")
            return None
    except Exception as e:
        print(f"❌ Error starting Celery worker: {e}")
        return None

def start_celery_beat():
    """Start Celery beat scheduler."""
    print("⏰ Starting Celery beat scheduler...")
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "celery", "-A", "tasks.task_queue", 
            "beat", "--loglevel=info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if process.poll() is None:
            print("✅ Celery beat started")
            return process
        else:
            print("❌ Failed to start Celery beat")
            return None
    except Exception as e:
        print(f"❌ Error starting Celery beat: {e}")
        return None

def find_available_port(start_port, max_attempts, logger):
    last_error = None
    for attempt in range(max_attempts):
        port = start_port + attempt
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                s.close()
                if attempt > 0:
                    logger.info(f"Port {start_port} in use, trying {port}")
                logger.info(f"Selected port {port} for API server.")
                return port
            except OSError as e:
                logger.warning(f"Port {port} is in use.")
                last_error = e
    logger.error(f"No available port found in range {start_port}-{start_port+max_attempts-1}")
    import sys
    sys.exit(1)

def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting MeRNSTA API server...")
    logger = get_logger("api")
    try:
        port = find_available_port(api_port, port_retry_attempts, logger)
        logger.info(f"Starting API server on port {port}")
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if process.poll() is None:
            print(f"✅ API server started at http://localhost:{port}")
            return process
        else:
            print("❌ Failed to start API server")
            return None
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        print(f"❌ Failed to start API server: {e}")
        import sys
        sys.exit(1)

def check_health():
    """Check system health."""
    print("🏥 Checking system health...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ System is healthy")
            return True
        else:
            print(f"⚠️  Health check returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def show_dashboard_info():
    """Show dashboard and monitoring information."""
    print("""
    📊 Monitoring & Dashboards:
    
    🔍 Health Checks:
       • http://localhost:8000/health          (Comprehensive health)
       • http://localhost:8000/health/live     (Liveness probe)
       • http://localhost:8000/health/ready    (Readiness probe)
       • http://localhost:8000/health/detailed (Detailed metrics)
    
    📈 Metrics:
       • http://localhost:8000/metrics         (Prometheus metrics)
    
    📚 API Documentation:
       • http://localhost:8000/docs            (Interactive API docs)
       • http://localhost:8000/redoc           (ReDoc documentation)
    
    🔧 Management:
       • Check logs: tail -f logs/mernsta.log
       • Cache stats: curl http://localhost:8000/api/memory/cache-stats
       • Task queue: curl http://localhost:8000/health/detailed
    """)

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Shutting down MeRNSTA Enterprise...")
    sys.exit(0)

def main():
    """Main startup function."""
    print_banner()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("💡 Please configure your environment and try again")
        sys.exit(1)
    
    # Check Redis
    if not start_redis():
        print("💡 Please start Redis and try again")
        sys.exit(1)
    
    # Start components
    processes = []
    
    # Start Celery worker
    worker_process = start_celery_worker()
    if worker_process:
        processes.append(worker_process)
    
    # Start Celery beat
    beat_process = start_celery_beat()
    if beat_process:
        processes.append(beat_process)
    
    # Start API server
    api_process = start_api_server()
    if api_process:
        processes.append(api_process)
    
    if not processes:
        print("❌ Failed to start any components")
        sys.exit(1)
    
    # Wait a moment for services to start
    time.sleep(5)
    
    # Check health
    if check_health():
        show_dashboard_info()
        
        print("\n🎉 MeRNSTA Enterprise is running!")
        print("💡 Press Ctrl+C to stop all services")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    else:
        print("❌ System health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 