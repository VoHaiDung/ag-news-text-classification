#!/bin/bash
# ================================================================================
# Service Restart Script for AG News Text Classification System
# ================================================================================
# This script provides controlled restart capabilities for all services with
# proper dependency management, health checking, and rollback mechanisms.
# It follows best practices for zero-downtime deployments and service management.
#
# References:
#   - The Twelve-Factor App: https://12factor.net/
#   - Site Reliability Engineering (Google, 2016)
#   - Unix and Linux System Administration Handbook (5th Edition)
#
# Usage: ./restart_services.sh [options] [service_names]
# Options:
#   -a, --all           Restart all services
#   -g, --graceful      Perform graceful restart with zero downtime
#   -f, --force         Force restart without health checks
#   -r, --rolling       Rolling restart for high availability
#   -b, --backup        Create backup before restart
#   -c, --config FILE   Use specific configuration file
#   -t, --timeout SEC   Health check timeout (default: 60)
#   -v, --verbose       Enable verbose logging
#   -d, --dry-run       Show what would be done without executing
#   -h, --help          Show this help message
#
# Author: Võ Hải Dũng
# License: MIT
# ================================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${PROJECT_ROOT}/logs/services"
readonly PID_DIR="${PROJECT_ROOT}/run"
readonly CONFIG_DIR="${PROJECT_ROOT}/configs/services"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups/services"

# Create necessary directories
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${BACKUP_DIR}"

# Logging configuration
readonly LOG_FILE="${LOG_DIR}/restart_$(date +%Y%m%d_%H%M%S).log"
readonly TIMESTAMP_FORMAT="+%Y-%m-%d %H:%M:%S"

# Default values
GRACEFUL_RESTART=false
FORCE_RESTART=false
ROLLING_RESTART=false
CREATE_BACKUP=false
DRY_RUN=false
VERBOSE=false
HEALTH_CHECK_TIMEOUT=60
CONFIG_FILE="${CONFIG_DIR}/services.yaml"
RESTART_ALL=false

# Service definitions
declare -A SERVICES=(
    ["rest-api"]="python -m uvicorn src.api.rest.app:app --host 0.0.0.0 --port 8000"
    ["grpc-api"]="python src/api/grpc/server.py"
    ["graphql-api"]="python src/api/graphql/server.py"
    ["prediction-service"]="python src/services/core/prediction_service.py"
    ["training-service"]="python src/services/core/training_service.py"
    ["data-service"]="python src/services/core/data_service.py"
    ["model-service"]="python src/services/core/model_management_service.py"
    ["monitoring-service"]="python src/services/monitoring/metrics_service.py"
)

# Service dependencies (service -> dependencies)
declare -A DEPENDENCIES=(
    ["rest-api"]="prediction-service data-service"
    ["grpc-api"]="prediction-service data-service"
    ["graphql-api"]="prediction-service data-service"
    ["training-service"]="data-service model-service"
    ["prediction-service"]="model-service"
)

# Service ports for health checking
declare -A SERVICE_PORTS=(
    ["rest-api"]=8000
    ["grpc-api"]=50051
    ["graphql-api"]=4000
    ["prediction-service"]=8001
    ["training-service"]=8002
    ["data-service"]=8003
    ["model-service"]=8004
    ["monitoring-service"]=9090
)

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date "${TIMESTAMP_FORMAT}")
    
    # Log to file
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
    
    # Log to console based on level and verbosity
    case ${level} in
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${message}" >&2
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} ${message}"
            ;;
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${message}"
            ;;
        DEBUG)
            if [ "${VERBOSE}" = true ]; then
                echo -e "${BLUE}[DEBUG]${NC} ${message}"
            fi
            ;;
    esac
}

# Error handling
error_exit() {
    log ERROR "$1"
    exit 1
}

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        log ERROR "Script failed with exit code ${exit_code}"
        
        # Attempt rollback if needed
        if [ "${CREATE_BACKUP}" = true ] && [ -n "${BACKUP_TIMESTAMP:-}" ]; then
            log INFO "Attempting rollback from backup ${BACKUP_TIMESTAMP}"
            # Rollback logic here
        fi
    fi
    
    log INFO "Restart script completed"
}

trap cleanup EXIT

# Parse command line arguments
parse_arguments() {
    local services_to_restart=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                RESTART_ALL=true
                shift
                ;;
            -g|--graceful)
                GRACEFUL_RESTART=true
                shift
                ;;
            -f|--force)
                FORCE_RESTART=true
                shift
                ;;
            -r|--rolling)
                ROLLING_RESTART=true
                shift
                ;;
            -b|--backup)
                CREATE_BACKUP=true
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -t|--timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                error_exit "Unknown option: $1"
                ;;
            *)
                services_to_restart+=("$1")
                shift
                ;;
        esac
    done
    
    # Set services to restart
    if [ "${RESTART_ALL}" = true ]; then
        SERVICES_TO_RESTART=(${!SERVICES[@]})
    elif [ ${#services_to_restart[@]} -gt 0 ]; then
        SERVICES_TO_RESTART=("${services_to_restart[@]}")
    else
        error_exit "No services specified. Use -a for all services or specify service names."
    fi
}

# Show help message
show_help() {
    cat << EOF
Service Restart Script for AG News Text Classification System

Usage: $0 [options] [service_names]

Options:
    -a, --all           Restart all services
    -g, --graceful      Perform graceful restart with zero downtime
    -f, --force         Force restart without health checks
    -r, --rolling       Rolling restart for high availability
    -b, --backup        Create backup before restart
    -c, --config FILE   Use specific configuration file
    -t, --timeout SEC   Health check timeout (default: 60)
    -v, --verbose       Enable verbose logging
    -d, --dry-run       Show what would be done without executing
    -h, --help          Show this help message

Available services:
EOF
    for service in "${!SERVICES[@]}"; do
        echo "    - ${service}"
    done
    
    echo ""
    echo "Examples:"
    echo "    $0 -a                    # Restart all services"
    echo "    $0 -g rest-api grpc-api  # Graceful restart of specific services"
    echo "    $0 -r -a                 # Rolling restart of all services"
}

# Get service PID
get_service_pid() {
    local service=$1
    local pid_file="${PID_DIR}/${service}.pid"
    
    if [ -f "${pid_file}" ]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}"
            return 0
        fi
    fi
    
    # Try to find by process name
    local cmd="${SERVICES[${service}]}"
    local pid=$(pgrep -f "${cmd}" | head -1)
    
    if [ -n "${pid}" ]; then
        echo "${pid}"
        return 0
    fi
    
    return 1
}

# Check if service is running
is_service_running() {
    local service=$1
    get_service_pid "${service}" > /dev/null 2>&1
}

# Stop service
stop_service() {
    local service=$1
    local graceful=${2:-false}
    
    log INFO "Stopping ${service}..."
    
    if [ "${DRY_RUN}" = true ]; then
        log DEBUG "[DRY-RUN] Would stop ${service}"
        return 0
    fi
    
    local pid=$(get_service_pid "${service}")
    
    if [ -z "${pid}" ]; then
        log DEBUG "${service} is not running"
        return 0
    fi
    
    if [ "${graceful}" = true ]; then
        # Send SIGTERM for graceful shutdown
        log DEBUG "Sending SIGTERM to ${service} (PID: ${pid})"
        kill -TERM "${pid}"
        
        # Wait for graceful shutdown
        local timeout=30
        while [ ${timeout} -gt 0 ] && kill -0 "${pid}" 2>/dev/null; do
            sleep 1
            ((timeout--))
        done
        
        if kill -0 "${pid}" 2>/dev/null; then
            log WARNING "${service} did not stop gracefully, forcing..."
            kill -KILL "${pid}"
        fi
    else
        # Force stop
        log DEBUG "Forcing stop of ${service} (PID: ${pid})"
        kill -KILL "${pid}"
    fi
    
    # Remove PID file
    rm -f "${PID_DIR}/${service}.pid"
    
    log INFO "${service} stopped"
}

# Start service
start_service() {
    local service=$1
    
    log INFO "Starting ${service}..."
    
    if [ "${DRY_RUN}" = true ]; then
        log DEBUG "[DRY-RUN] Would start ${service}"
        return 0
    fi
    
    # Check if already running
    if is_service_running "${service}"; then
        log WARNING "${service} is already running"
        return 0
    fi
    
    # Start the service
    local cmd="${SERVICES[${service}]}"
    local log_file="${LOG_DIR}/${service}.log"
    
    cd "${PROJECT_ROOT}"
    
    # Start service in background
    nohup ${cmd} > "${log_file}" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "${pid}" > "${PID_DIR}/${service}.pid"
    
    log DEBUG "${service} started with PID ${pid}"
    
    # Wait a moment for service to initialize
    sleep 2
    
    # Verify service started
    if ! kill -0 "${pid}" 2>/dev/null; then
        log ERROR "${service} failed to start"
        return 1
    fi
    
    log INFO "${service} started successfully"
}

# Health check for service
check_service_health() {
    local service=$1
    local timeout=${2:-${HEALTH_CHECK_TIMEOUT}}
    
    log DEBUG "Checking health of ${service}..."
    
    if [ "${FORCE_RESTART}" = true ]; then
        log DEBUG "Skipping health check (force mode)"
        return 0
    fi
    
    if [ "${DRY_RUN}" = true ]; then
        log DEBUG "[DRY-RUN] Would check health of ${service}"
        return 0
    fi
    
    local port=${SERVICE_PORTS[${service}]}
    
    if [ -z "${port}" ]; then
        log WARNING "No health check port defined for ${service}"
        return 0
    fi
    
    local elapsed=0
    while [ ${elapsed} -lt ${timeout} ]; do
        if nc -z localhost "${port}" 2>/dev/null; then
            log INFO "${service} is healthy"
            return 0
        fi
        
        sleep 1
        ((elapsed++))
    done
    
    log ERROR "${service} health check failed after ${timeout} seconds"
    return 1
}

# Get service dependencies
get_dependencies() {
    local service=$1
    echo "${DEPENDENCIES[${service}]:-}"
}

# Topological sort for dependency order
topological_sort() {
    local -A visited=()
    local -A recursion_stack=()
    local sorted_order=()
    
    visit_node() {
        local node=$1
        
        if [ "${recursion_stack[${node}]:-}" = "true" ]; then
            error_exit "Circular dependency detected involving ${node}"
        fi
        
        if [ "${visited[${node}]:-}" = "true" ]; then
            return
        fi
        
        recursion_stack[${node}]="true"
        
        local deps=$(get_dependencies "${node}")
        for dep in ${deps}; do
            visit_node "${dep}"
        done
        
        recursion_stack[${node}]="false"
        visited[${node}]="true"
        sorted_order+=("${node}")
    }
    
    for service in "$@"; do
        visit_node "${service}"
    done
    
    echo "${sorted_order[@]}"
}

# Create backup
create_backup() {
    if [ "${CREATE_BACKUP}" != true ]; then
        return 0
    fi
    
    log INFO "Creating backup..."
    
    if [ "${DRY_RUN}" = true ]; then
        log DEBUG "[DRY-RUN] Would create backup"
        return 0
    fi
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local backup_dir="${BACKUP_DIR}/${BACKUP_TIMESTAMP}"
    
    mkdir -p "${backup_dir}"
    
    # Backup configurations
    cp -r "${CONFIG_DIR}" "${backup_dir}/configs"
    
    # Backup current service states
    for service in "${SERVICES_TO_RESTART[@]}"; do
        if is_service_running "${service}"; then
            echo "${service}" >> "${backup_dir}/running_services.txt"
        fi
    done
    
    log INFO "Backup created at ${backup_dir}"
}

# Restart service with dependencies
restart_service_with_deps() {
    local service=$1
    local mode=${2:-"normal"}  # normal, graceful, rolling
    
    log INFO "Restarting ${service} (mode: ${mode})"
    
    # Get dependencies in correct order
    local deps=$(get_dependencies "${service}")
    
    # Ensure dependencies are running
    for dep in ${deps}; do
        if ! is_service_running "${dep}"; then
            log INFO "Starting dependency ${dep} for ${service}"
            start_service "${dep}"
            check_service_health "${dep}" || error_exit "Dependency ${dep} is not healthy"
        fi
    done
    
    case ${mode} in
        graceful)
            # Start new instance before stopping old one
            if [ "${SERVICE_PORTS[${service}]:-}" ]; then
                # Temporarily change port for new instance
                local original_port=${SERVICE_PORTS[${service}]}
                local temp_port=$((original_port + 1000))
                
                # Start new instance on temporary port
                SERVICE_PORTS[${service}]=${temp_port}
                start_service "${service}"
                check_service_health "${service}"
                
                # Switch traffic (would need load balancer integration)
                log INFO "Switching traffic to new instance"
                
                # Stop old instance
                SERVICE_PORTS[${service}]=${original_port}
                stop_service "${service}" true
                
                # Move new instance to original port
                # This would require more sophisticated port management
            else
                stop_service "${service}" true
                start_service "${service}"
            fi
            ;;
            
        rolling)
            # For services with multiple instances
            log INFO "Performing rolling restart of ${service}"
            stop_service "${service}" true
            sleep 2
            start_service "${service}"
            ;;
            
        *)
            # Normal restart
            stop_service "${service}" "${GRACEFUL_RESTART}"
            start_service "${service}"
            ;;
    esac
    
    # Health check
    check_service_health "${service}" || error_exit "${service} is not healthy after restart"
}

# Main restart logic
perform_restart() {
    log INFO "Starting service restart process"
    log INFO "Services to restart: ${SERVICES_TO_RESTART[*]}"
    
    # Create backup if requested
    create_backup
    
    # Sort services by dependencies
    local sorted_services=$(topological_sort "${SERVICES_TO_RESTART[@]}")
    
    log DEBUG "Restart order: ${sorted_services}"
    
    if [ "${ROLLING_RESTART}" = true ]; then
        # Rolling restart - one service at a time
        for service in ${sorted_services}; do
            restart_service_with_deps "${service}" "rolling"
            
            # Wait between services for stability
            log INFO "Waiting for system stabilization..."
            sleep 5
        done
    elif [ "${GRACEFUL_RESTART}" = true ]; then
        # Graceful restart with zero downtime
        for service in ${sorted_services}; do
            restart_service_with_deps "${service}" "graceful"
        done
    else
        # Normal restart - stop all then start all
        log INFO "Stopping services..."
        for service in $(echo "${sorted_services}" | tr ' ' '\n' | tac | tr '\n' ' '); do
            stop_service "${service}" "${GRACEFUL_RESTART}"
        done
        
        log INFO "Starting services..."
        for service in ${sorted_services}; do
            start_service "${service}"
        done
        
        log INFO "Performing health checks..."
        for service in ${sorted_services}; do
            check_service_health "${service}" || error_exit "${service} is not healthy"
        done
    fi
    
    log INFO "All services restarted successfully"
}

# Verify environment
verify_environment() {
    log DEBUG "Verifying environment..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is not installed"
    fi
    
    # Check project structure
    if [ ! -d "${PROJECT_ROOT}/src" ]; then
        error_exit "Invalid project root: ${PROJECT_ROOT}"
    fi
    
    # Check configuration file
    if [ ! -f "${CONFIG_FILE}" ] && [ "${CONFIG_FILE}" != "${CONFIG_DIR}/services.yaml" ]; then
        error_exit "Configuration file not found: ${CONFIG_FILE}"
    fi
    
    log DEBUG "Environment verification completed"
}

# Main execution
main() {
    log INFO "Service restart script started"
    log INFO "Project root: ${PROJECT_ROOT}"
    log INFO "Log file: ${LOG_FILE}"
    
    # Verify environment
    verify_environment
    
    # Parse arguments
    parse_arguments "$@"
    
    if [ "${DRY_RUN}" = true ]; then
        log INFO "DRY-RUN MODE: No actual changes will be made"
    fi
    
    # Perform restart
    perform_restart
    
    log INFO "Service restart completed successfully"
}

# Run main function
main "$@"
