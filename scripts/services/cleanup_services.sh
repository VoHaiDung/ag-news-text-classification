#!/bin/bash
# ================================================================================
# Service Cleanup Script for AG News Text Classification System
# ================================================================================
# This script performs cleanup operations for all services including stopping
# processes, removing temporary files, clearing caches, and releasing resources.
# It follows best practices for graceful shutdown and resource management.
#
# References:
#   - Linux System Administration Handbook (Nemeth et al., 2017)
#   - Site Reliability Engineering (Google, 2016)
#   - Production-Ready Microservices (O'Reilly, 2016)
#
# Author: Võ Hải Dũng
# License: MIT
# ================================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
CACHE_DIR="${PROJECT_ROOT}/cache"
TMP_DIR="${PROJECT_ROOT}/tmp"
PID_DIR="${PROJECT_ROOT}/run"

# Service configuration
SERVICES=(
    "rest-api:8000"
    "grpc-server:50051"
    "graphql-server:4000"
    "prediction-service:8001"
    "training-service:8002"
    "data-service:8003"
    "evaluation-service:8004"
    "monitoring-service:9090"
)

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Function to check if process is running
is_process_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

# Function to stop service gracefully
stop_service() {
    local service_name=$1
    local port=$2
    local pid_file="${PID_DIR}/${service_name}.pid"
    
    log_info "Stopping ${service_name}..."
    
    # Try to read PID from file
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if is_process_running "$pid"; then
            # Send SIGTERM for graceful shutdown
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait for process to stop (max 10 seconds)
            local count=0
            while is_process_running "$pid" && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if is_process_running "$pid"; then
                log_warning "Force killing ${service_name} (PID: $pid)"
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            log_info "${service_name} stopped"
        else
            log_info "${service_name} not running (stale PID file)"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        # Try to find process by port
        local pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            log_info "Found ${service_name} on port ${port} (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            
            # Force kill if still running
            if lsof -ti:$port >/dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            log_info "${service_name} stopped"
        else
            log_info "${service_name} not running"
        fi
    fi
}

# Function to cleanup Docker containers
cleanup_docker() {
    log_section "Cleaning up Docker containers"
    
    # Stop AG News related containers
    local containers=$(docker ps -q --filter "label=app=agnews" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        log_info "Stopping Docker containers..."
        docker stop $containers || true
        docker rm $containers || true
        log_info "Docker containers cleaned up"
    else
        log_info "No Docker containers to clean"
    fi
    
    # Remove dangling images
    local dangling=$(docker images -q --filter "dangling=true" 2>/dev/null || true)
    if [ -n "$dangling" ]; then
        log_info "Removing dangling Docker images..."
        docker rmi $dangling || true
    fi
}

# Function to cleanup Kubernetes resources
cleanup_kubernetes() {
    log_section "Cleaning up Kubernetes resources"
    
    if command -v kubectl &> /dev/null; then
        # Check if namespace exists
        if kubectl get namespace agnews-api &> /dev/null; then
            log_info "Deleting Kubernetes resources in agnews-api namespace..."
            
            # Delete deployments
            kubectl delete deployments --all -n agnews-api --grace-period=30 || true
            
            # Delete services
            kubectl delete services --all -n agnews-api || true
            
            # Delete pods
            kubectl delete pods --all -n agnews-api --grace-period=10 || true
            
            # Delete configmaps and secrets
            kubectl delete configmaps --all -n agnews-api || true
            kubectl delete secrets --all -n agnews-api || true
            
            log_info "Kubernetes resources cleaned up"
        else
            log_info "Kubernetes namespace agnews-api not found"
        fi
    else
        log_info "kubectl not found, skipping Kubernetes cleanup"
    fi
}

# Function to cleanup cache files
cleanup_cache() {
    log_section "Cleaning up cache"
    
    # Redis cache
    if command -v redis-cli &> /dev/null; then
        log_info "Flushing Redis cache..."
        redis-cli FLUSHDB 2>/dev/null || log_warning "Could not flush Redis cache"
    fi
    
    # File-based cache
    if [ -d "$CACHE_DIR" ]; then
        log_info "Removing cache directory: $CACHE_DIR"
        rm -rf "$CACHE_DIR"/*
        log_info "Cache directory cleaned"
    fi
    
    # Python cache
    log_info "Removing Python cache files..."
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Model cache
    local model_cache="${PROJECT_ROOT}/.cache/huggingface"
    if [ -d "$model_cache" ]; then
        log_info "Cleaning model cache..."
        rm -rf "$model_cache"/*
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    log_section "Cleaning up temporary files"
    
    # Remove temporary directory
    if [ -d "$TMP_DIR" ]; then
        log_info "Removing temporary directory: $TMP_DIR"
        rm -rf "$TMP_DIR"/*
        log_info "Temporary files cleaned"
    fi
    
    # Remove lock files
    log_info "Removing lock files..."
    find "$PROJECT_ROOT" -name "*.lock" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name ".~lock*" -type f -delete 2>/dev/null || true
    
    # Remove backup files
    log_info "Removing backup files..."
    find "$PROJECT_ROOT" -name "*~" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.bak" -type f -delete 2>/dev/null || true
}

# Function to cleanup log files
cleanup_logs() {
    log_section "Cleaning up log files"
    
    if [ -d "$LOG_DIR" ]; then
        # Archive old logs
        local archive_dir="${LOG_DIR}/archive"
        mkdir -p "$archive_dir"
        
        # Compress logs older than 7 days
        log_info "Archiving old log files..."
        find "$LOG_DIR" -name "*.log" -type f -mtime +7 -exec gzip {} \; 2>/dev/null || true
        find "$LOG_DIR" -name "*.log.gz" -type f -exec mv {} "$archive_dir/" \; 2>/dev/null || true
        
        # Remove logs older than 30 days
        log_info "Removing very old log files..."
        find "$archive_dir" -name "*.log.gz" -type f -mtime +30 -delete 2>/dev/null || true
        
        # Truncate current log files if they're too large (> 100MB)
        for logfile in "$LOG_DIR"/*.log; do
            if [ -f "$logfile" ]; then
                size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null || echo 0)
                if [ "$size" -gt 104857600 ]; then
                    log_info "Truncating large log file: $(basename "$logfile")"
                    echo "Log truncated at $(date)" > "$logfile"
                fi
            fi
        done
        
        log_info "Log files cleaned"
    fi
}

# Function to cleanup database connections
cleanup_database() {
    log_section "Cleaning up database connections"
    
    # PostgreSQL
    if command -v psql &> /dev/null; then
        log_info "Terminating PostgreSQL connections..."
        psql -U postgres -c "
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = 'agnews'
            AND pid <> pg_backend_pid()
        " 2>/dev/null || log_warning "Could not terminate PostgreSQL connections"
    fi
    
    # MongoDB
    if command -v mongo &> /dev/null; then
        log_info "Closing MongoDB connections..."
        mongo --eval "db.adminCommand({killAllSessions: []})" 2>/dev/null || \
            log_warning "Could not close MongoDB connections"
    fi
}

# Function to cleanup message queues
cleanup_queues() {
    log_section "Cleaning up message queues"
    
    # RabbitMQ
    if command -v rabbitmqctl &> /dev/null; then
        log_info "Purging RabbitMQ queues..."
        rabbitmqctl list_queues name | tail -n +2 | while read queue; do
            rabbitmqctl purge_queue "$queue" 2>/dev/null || true
        done
    fi
    
    # Celery
    if pgrep -f celery > /dev/null; then
        log_info "Stopping Celery workers..."
        pkill -f celery || true
    fi
}

# Function to release GPU resources
cleanup_gpu() {
    log_section "Cleaning up GPU resources"
    
    if command -v nvidia-smi &> /dev/null; then
        log_info "Checking GPU processes..."
        
        # Get PIDs using GPU
        local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
        
        if [ -n "$gpu_pids" ]; then
            log_info "Terminating GPU processes..."
            for pid in $gpu_pids; do
                if ps -p "$pid" > /dev/null 2>&1; then
                    local cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
                    if [[ "$cmd" == *"python"* ]] || [[ "$cmd" == *"agnews"* ]]; then
                        log_info "Killing GPU process: PID=$pid CMD=$cmd"
                        kill -TERM "$pid" 2>/dev/null || true
                    fi
                fi
            done
            
            sleep 2
            
            # Force kill if still running
            for pid in $gpu_pids; do
                if ps -p "$pid" > /dev/null 2>&1; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            done
            
            log_info "GPU resources released"
        else
            log_info "No GPU processes to clean"
        fi
        
        # Reset GPU if needed
        if [ "$FORCE_GPU_RESET" = "true" ]; then
            log_warning "Resetting GPU..."
            nvidia-smi --gpu-reset 2>/dev/null || log_error "Could not reset GPU"
        fi
    else
        log_info "No NVIDIA GPU detected"
    fi
}

# Function to cleanup network resources
cleanup_network() {
    log_section "Cleaning up network resources"
    
    # Close hanging connections
    log_info "Checking for hanging connections..."
    
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        # Check for CLOSE_WAIT connections
        local close_wait=$(netstat -an | grep ":$port" | grep CLOSE_WAIT | wc -l)
        if [ "$close_wait" -gt 0 ]; then
            log_warning "Found $close_wait CLOSE_WAIT connections on port $port"
        fi
    done
    
    # Clean up iptables rules (if any custom rules were added)
    if [ "$CLEANUP_IPTABLES" = "true" ]; then
        log_info "Cleaning iptables rules..."
        iptables -t nat -F 2>/dev/null || true
        iptables -F 2>/dev/null || true
    fi
}

# Function to perform health check after cleanup
post_cleanup_check() {
    log_section "Post-cleanup verification"
    
    local issues=0
    
    # Check if services are stopped
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        if lsof -ti:$port >/dev/null 2>&1; then
            log_error "Service still running on port $port"
            issues=$((issues + 1))
        fi
    done
    
    # Check for remaining processes
    if pgrep -f "agnews" > /dev/null 2>&1; then
        log_warning "Found remaining AG News processes"
        pgrep -f "agnews" -l
        issues=$((issues + 1))
    fi
    
    # Check disk space
    local disk_usage=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 90 ]; then
        log_info "Disk usage: ${disk_usage}%"
    else
        log_warning "High disk usage: ${disk_usage}%"
        issues=$((issues + 1))
    fi
    
    if [ $issues -eq 0 ]; then
        log_info "✓ Cleanup completed successfully"
        return 0
    else
        log_warning "⚠ Cleanup completed with $issues issues"
        return 1
    fi
}

# Main cleanup function
main() {
    log_section "AG News Service Cleanup"
    log_info "Starting cleanup at $(date)"
    
    # Parse arguments
    FORCE_GPU_RESET=false
    CLEANUP_IPTABLES=false
    SKIP_DOCKER=false
    SKIP_K8S=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force-gpu-reset)
                FORCE_GPU_RESET=true
                shift
                ;;
            --cleanup-iptables)
                CLEANUP_IPTABLES=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-k8s)
                SKIP_K8S=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --force-gpu-reset    Force GPU reset"
                echo "  --cleanup-iptables   Clean iptables rules"
                echo "  --skip-docker        Skip Docker cleanup"
                echo "  --skip-k8s          Skip Kubernetes cleanup"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Stop all services
    log_section "Stopping services"
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        stop_service "$service" "$port"
    done
    
    # Cleanup Docker
    if [ "$SKIP_DOCKER" = false ]; then
        cleanup_docker
    fi
    
    # Cleanup Kubernetes
    if [ "$SKIP_K8S" = false ]; then
        cleanup_kubernetes
    fi
    
    # Cleanup resources
    cleanup_cache
    cleanup_temp_files
    cleanup_logs
    cleanup_database
    cleanup_queues
    cleanup_gpu
    cleanup_network
    
    # Post-cleanup verification
    post_cleanup_check
    
    log_section "Cleanup Complete"
    log_info "Finished at $(date)"
}

# Run main function
main "$@"
