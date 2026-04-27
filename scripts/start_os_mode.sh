#!/bin/bash
# MeRNSTA OS Integration Startup Script - Phase 30
# Launcher for integration runner, shell, or API server with environment detection

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
MODE="daemon"
COMPONENT=""
API_HOST="127.0.0.1"
API_PORT="8181"
DAEMON_MODE="daemon"
VERBOSE=false
BACKGROUND=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_info() {
    print_color "$BLUE" "ℹ️  $1"
}

print_success() {
    print_color "$GREEN" "✅ $1"
}

print_warning() {
    print_color "$YELLOW" "⚠️  $1"
}

print_error() {
    print_color "$RED" "❌ $1"
}

print_header() {
    echo
    print_color "$CYAN" "🧠 ================================================"
    print_color "$CYAN" "   MeRNSTA OS Integration - Phase 30"
    print_color "$CYAN" "   $1"
    print_color "$CYAN" "================================================"
    echo
}

# Function to detect operating system
detect_os() {
    if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
        echo "WSL"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "Cygwin"
    elif [[ "$OSTYPE" == "msys" ]]; then
        echo "MinGW"
    else
        echo "Unknown"
    fi
}

# Function to check if Python is available
check_python() {
    local python_cmd=""
    
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    elif command -v python &> /dev/null; then
        python_cmd="python"
    else
        print_error "Python not found. Please install Python 3.7+."
        exit 1
    fi
    
    # Check Python version
    local python_version=$($python_cmd -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    local major_version=$(echo $python_version | cut -d. -f1)
    local minor_version=$(echo $python_version | cut -d. -f2)
    
    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 7 ]]; then
        print_error "Python 3.7+ required. Found: $python_version"
        exit 1
    fi
    
    echo $python_cmd
}

# Function to check if virtual environment exists and activate it
setup_venv() {
    local python_cmd=$1
    
    if [[ -d "$PROJECT_ROOT/.venv" ]]; then
        print_info "Activating virtual environment..."
        source "$PROJECT_ROOT/.venv/bin/activate"
        return 0
    elif [[ -d "$PROJECT_ROOT/venv" ]]; then
        print_info "Activating virtual environment..."
        source "$PROJECT_ROOT/venv/bin/activate"
        return 0
    fi
    
    print_warning "No virtual environment found. Using system Python."
    return 1
}

# Function to check dependencies
check_dependencies() {
    local python_cmd=$1
    
    print_info "Checking dependencies..."
    
    # Check if requirements.txt exists
    if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
        print_warning "requirements.txt not found. Continuing anyway..."
        return 0
    fi
    
    # Check key dependencies
    local missing_deps=()
    
    if ! $python_cmd -c "import fastapi" &> /dev/null; then
        missing_deps+=("fastapi")
    fi
    
    if ! $python_cmd -c "import uvicorn" &> /dev/null; then
        missing_deps+=("uvicorn")
    fi
    
    if ! $python_cmd -c "import aiohttp" &> /dev/null; then
        missing_deps+=("aiohttp")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        print_info "Install with: pip install -r requirements.txt"
        
        # Ask if user wants to install dependencies
        read -p "Install missing dependencies now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing dependencies..."
            $python_cmd -m pip install -r "$PROJECT_ROOT/requirements.txt"
            print_success "Dependencies installed"
        else
            print_warning "Continuing without installing dependencies..."
        fi
    fi
}

# Function to check if a process is running
is_process_running() {
    local process_name=$1
    pgrep -f "$process_name" > /dev/null 2>&1
}

# Function to start the integration runner
start_integration_runner() {
    local python_cmd=$1
    local mode=$2
    
    print_header "Starting MeRNSTA Integration Runner"
    print_info "Mode: $mode"
    print_info "OS: $(detect_os)"
    
    cd "$PROJECT_ROOT"
    
    local cmd="$python_cmd system/integration_runner.py --mode=$mode"
    
    if [[ $VERBOSE == true ]]; then
        print_info "Command: $cmd"
    fi
    
    if [[ $BACKGROUND == true ]]; then
        print_info "Starting in background..."
        nohup $cmd > "output/integration_runner.log" 2>&1 &
        local pid=$!
        echo $pid > "pids/integration_runner.pid"
        print_success "Integration runner started with PID: $pid"
        print_info "Log file: output/integration_runner.log"
    else
        print_info "Starting integration runner..."
        $cmd
    fi
}

# Function to start the API server
start_api_server() {
    local python_cmd=$1
    
    print_header "Starting MeRNSTA System Bridge API"
    print_info "Host: $API_HOST"
    print_info "Port: $API_PORT"
    
    cd "$PROJECT_ROOT"
    
    local cmd="$python_cmd api/system_bridge.py --host=$API_HOST --port=$API_PORT"
    
    if [[ $VERBOSE == true ]]; then
        print_info "Command: $cmd"
    fi
    
    if [[ $BACKGROUND == true ]]; then
        print_info "Starting API server in background..."
        nohup $cmd > "output/api_server.log" 2>&1 &
        local pid=$!
        echo $pid > "pids/api_server.pid"
        print_success "API server started with PID: $pid"
        print_info "API available at: http://$API_HOST:$API_PORT"
        print_info "Log file: output/api_server.log"
    else
        print_info "Starting API server..."
        $cmd
    fi
}

# Function to start the interactive shell
start_shell() {
    local python_cmd=$1
    
    print_header "Starting MeRNSTA Interactive Shell"
    print_info "Connecting to API at: $API_HOST:$API_PORT"
    
    cd "$PROJECT_ROOT"
    
    local cmd="$python_cmd cli/mernsta_shell.py --host=$API_HOST --port=$API_PORT"
    
    if [[ $VERBOSE == true ]]; then
        print_info "Command: $cmd"
    fi
    
    print_info "Starting interactive shell..."
    $cmd
}

# Function to start the visualizer web server
start_visualizer() {
    local python_cmd=$1
    
    print_header "Starting MeRNSTA Memory Graph Visualizer"
    
    # Check if visualizer is enabled in config
    if ! grep -q "enable_visualizer: true" configs/config.yaml 2>/dev/null; then
        print_error "Visualizer is not enabled in configs/config.yaml"
        print_info "Set 'visualizer.enable_visualizer: true' in configs/config.yaml to enable"
        exit 1
    fi
    
    # Get visualizer port from config (default to 8182)
    local vis_port=$(grep -A 10 "^visualizer:" configs/config.yaml | grep "port:" | awk '{print $2}' | head -1 | tr -d '"')
    vis_port=${vis_port:-8182}
    
    # Get visualizer host from config (default to 127.0.0.1)
    local vis_host=$(grep -A 10 "^visualizer:" configs/config.yaml | grep "host:" | awk '{print $2}' | head -1 | tr -d '"')
    vis_host=${vis_host:-127.0.0.1}
    
    print_info "Host: $vis_host"
    print_info "Port: $vis_port"
    print_info "Visualizer URL: http://$vis_host:$vis_port/visualizer/"
    
    cd "$PROJECT_ROOT"
    
    # Start the web server with visualizer routes
    local cmd="$python_cmd -m uvicorn web.main:app --host=$vis_host --port=$vis_port --reload"
    
    if [[ $VERBOSE == true ]]; then
        print_info "Command: $cmd"
    fi
    
    if [[ $BACKGROUND == true ]]; then
        print_info "Starting visualizer in background..."
        nohup $cmd > "output/visualizer.log" 2>&1 &
        local pid=$!
        echo $pid > "pids/visualizer.pid"
        print_success "Visualizer started with PID: $pid"
        print_info "Visualizer available at: http://$vis_host:$vis_port/visualizer/"
        print_info "Log file: output/visualizer.log"
        print_warning "Note: Make sure the System Bridge API is running on port 8181"
    else
        print_info "Starting visualizer web server..."
        print_warning "Note: Make sure the System Bridge API is running on port 8181"
        $cmd
    fi
}

# Function to stop running processes
stop_processes() {
    print_header "Stopping MeRNSTA OS Integration Processes"
    
    # Create pids directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/pids"
    
    # Stop integration runner
    if [[ -f "$PROJECT_ROOT/pids/integration_runner.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/pids/integration_runner.pid")
        if kill -0 "$pid" 2> /dev/null; then
            print_info "Stopping integration runner (PID: $pid)..."
            kill "$pid"
            rm -f "$PROJECT_ROOT/pids/integration_runner.pid"
            print_success "Integration runner stopped"
        else
            print_warning "Integration runner process not found"
            rm -f "$PROJECT_ROOT/pids/integration_runner.pid"
        fi
    fi
    
    # Stop API server
    if [[ -f "$PROJECT_ROOT/pids/api_server.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/pids/api_server.pid")
        if kill -0 "$pid" 2> /dev/null; then
            print_info "Stopping API server (PID: $pid)..."
            kill "$pid"
            rm -f "$PROJECT_ROOT/pids/api_server.pid"
            print_success "API server stopped"
        else
            print_warning "API server process not found"
            rm -f "$PROJECT_ROOT/pids/api_server.pid"
        fi
    fi
    
    # Stop Visualizer
    if [[ -f "$PROJECT_ROOT/pids/visualizer.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/pids/visualizer.pid")
        if kill -0 "$pid" 2> /dev/null; then
            print_info "Stopping visualizer (PID: $pid)..."
            kill "$pid"
            rm -f "$PROJECT_ROOT/pids/visualizer.pid"
            print_success "Visualizer stopped"
        else
            print_warning "Visualizer process not found"
            rm -f "$PROJECT_ROOT/pids/visualizer.pid"
        fi
    fi
    
    # Also try to kill by process name as fallback
    if pgrep -f "integration_runner.py" > /dev/null; then
        print_info "Force stopping integration runner..."
        pkill -f "integration_runner.py"
    fi
    
    if pgrep -f "system_bridge.py" > /dev/null; then
        print_info "Force stopping API server..."
        pkill -f "system_bridge.py"
    fi
    
    if pgrep -f "web.main:app" > /dev/null; then
        print_info "Force stopping visualizer..."
        pkill -f "web.main:app"
    fi
    
    print_success "All processes stopped"
}

# Function to show status
show_status() {
    print_header "MeRNSTA OS Integration Status"
    
    print_info "Operating System: $(detect_os)"
    print_info "Project Root: $PROJECT_ROOT"
    
    echo
    print_info "Process Status:"
    
    # Check integration runner
    if [[ -f "$PROJECT_ROOT/pids/integration_runner.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/pids/integration_runner.pid")
        if kill -0 "$pid" 2> /dev/null; then
            print_success "Integration Runner: Running (PID: $pid)"
        else
            print_warning "Integration Runner: PID file exists but process not running"
        fi
    else
        print_warning "Integration Runner: Not running"
    fi
    
    # Check API server
    if [[ -f "$PROJECT_ROOT/pids/api_server.pid" ]]; then
        local pid=$(cat "$PROJECT_ROOT/pids/api_server.pid")
        if kill -0 "$pid" 2> /dev/null; then
            print_success "API Server: Running (PID: $pid)"
        else
            print_warning "API Server: PID file exists but process not running"
        fi
    else
        print_warning "API Server: Not running"
    fi
    
    # Check if API is responding
    echo
    print_info "API Health Check:"
    if command -v curl &> /dev/null; then
        if curl -s "http://$API_HOST:$API_PORT/health" > /dev/null 2>&1; then
            print_success "API is responding at http://$API_HOST:$API_PORT"
        else
            print_warning "API is not responding at http://$API_HOST:$API_PORT"
        fi
    else
        print_warning "curl not available for health check"
    fi
}

# Function to show help
show_help() {
    echo "🧠 MeRNSTA OS Integration Startup Script"
    echo
    echo "Usage: $0 [OPTIONS] COMPONENT"
    echo
    echo "COMPONENTS:"
    echo "  daemon                 Start the integration runner in daemon mode"
    echo "  api                    Start the API server only"
    echo "  shell                  Start the interactive shell"
    echo "  visualizer             Start the memory graph visualizer web interface"
    echo "  stop                   Stop all running processes"
    echo "  status                 Show status of running processes"
    echo "  help                   Show this help message"
    echo
    echo "OPTIONS:"
    echo "  --mode=MODE           Integration runner mode (daemon, interactive, headless, bridge_only)"
    echo "  --host=HOST           API server host (default: 127.0.0.1)"
    echo "  --port=PORT           API server port (default: 8181)"
    echo "  --background          Run in background (daemon/api only)"
    echo "  --verbose             Enable verbose output"
    echo
    echo "EXAMPLES:"
    echo "  $0 daemon                           # Start daemon mode"
    echo "  $0 api --background                 # Start API server in background"
    echo "  $0 shell                            # Start interactive shell"
    echo "  $0 visualizer                       # Start memory graph visualizer (port 8182)"
    echo "  $0 daemon --mode=interactive        # Start integration runner in interactive mode"
    echo "  $0 stop                             # Stop all processes"
    echo
    echo "ENVIRONMENT DETECTION:"
    echo "  Automatically detects WSL, macOS, Linux environments"
    echo "  Activates virtual environment if found (.venv or venv)"
    echo "  Checks and optionally installs dependencies"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*)
            DAEMON_MODE="${1#*=}"
            shift
            ;;
        --host=*)
            API_HOST="${1#*=}"
            shift
            ;;
        --port=*)
            API_PORT="${1#*=}"
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        daemon|api|shell|visualizer|stop|status|help)
            COMPONENT="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/output"
    mkdir -p "$PROJECT_ROOT/pids"
    
    # Handle component commands
    case $COMPONENT in
        "help")
            show_help
            exit 0
            ;;
        "stop")
            stop_processes
            exit 0
            ;;
        "status")
            show_status
            exit 0
            ;;
        "daemon"|"api"|"shell"|"visualizer")
            # Continue with setup
            ;;
        *)
            print_error "No component specified or unknown component: $COMPONENT"
            show_help
            exit 1
            ;;
    esac
    
    # System checks
    print_info "Detecting system environment..."
    local os_type=$(detect_os)
    print_success "Operating System: $os_type"
    
    # Check Python
    local python_cmd=$(check_python)
    print_success "Python: $python_cmd"
    
    # Setup virtual environment
    setup_venv "$python_cmd"
    
    # Check dependencies
    check_dependencies "$python_cmd"
    
    # Execute component
    case $COMPONENT in
        "daemon")
            start_integration_runner "$python_cmd" "$DAEMON_MODE"
            ;;
        "api")
            start_api_server "$python_cmd"
            ;;
        "shell")
            start_shell "$python_cmd"
            ;;
        "visualizer")
            start_visualizer "$python_cmd"
            ;;
    esac
}

# Run main function
main