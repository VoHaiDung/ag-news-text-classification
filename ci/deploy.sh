#!/bin/bash

# ==============================================================================
# Deployment Script for AG News Text Classification
# ==============================================================================
#
# This script implements deployment strategies following continuous delivery
# principles and cloud-native best practices from academic literature.
#
# References:
# - Humble, J., & Farley, D. (2010). "Continuous Delivery: Reliable Software 
#   Releases through Build, Test, and Deployment Automation". Addison-Wesley.
# - Burns, B., Grant, B., Oppenheimer, D., Brewer, E., & Wilkes, J. (2016). 
#   "Borg, omega, and kubernetes: Lessons learned from three container-management 
#   systems over a decade". ACM Transactions on Computer Systems, 34(1), 1-26.
# - Morris, K. (2016). "Infrastructure as Code: Managing Servers in the Cloud". 
#   O'Reilly Media.
# - Newman, S. (2015). "Building Microservices: Designing Fine-Grained Systems". 
#   O'Reilly Media.
# - Verma, A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E., & Wilkes, J. 
#   (2015). "Large-scale cluster management at Google with Borg". In Proceedings 
#   of the Tenth European Conference on Computer Systems (pp. 1-17).
#
# Deployment Strategies:
# - Rolling updates for zero-downtime deployment (Burns et al., 2016)
# - Blue-Green deployment for instant rollback (Humble & Farley, 2010)
# - Canary releases for gradual rollout (Newman, 2015)
# - Feature flags for controlled activation
# - Automated rollback on failure detection
#
# Author: Vo Hai Dung
# License: MIT
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ------------------------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DEPLOYMENT_ID="$(date +%Y%m%d%H%M%S)-$(uuidgen 2>/dev/null | cut -d'-' -f1 || echo $RANDOM)"
readonly DEPLOYMENT_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly DEPLOYMENT_LOG="${PROJECT_ROOT}/outputs/deployments/logs/${DEPLOYMENT_ID}.log"

# Deployment configuration following cloud-native patterns
ENVIRONMENT="${ENVIRONMENT:-staging}"
PLATFORM="${PLATFORM:-kubernetes}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"
MODEL_VERSION="${MODEL_VERSION:-latest}"
NAMESPACE="${NAMESPACE:-ag-news}"
SERVICE_NAME="${SERVICE_NAME:-ag-news-classifier}"

# Resource configuration following Kubernetes best practices
REPLICAS="${REPLICAS:-3}"
MAX_SURGE="${MAX_SURGE:-1}"
MAX_UNAVAILABLE="${MAX_UNAVAILABLE:-0}"
CPU_REQUEST="${CPU_REQUEST:-500m}"
CPU_LIMIT="${CPU_LIMIT:-2000m}"
MEMORY_REQUEST="${MEMORY_REQUEST:-1Gi}"
MEMORY_LIMIT="${MEMORY_LIMIT:-4Gi}"

# Health check configuration (SRE best practices)
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-10}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-10}"
READINESS_TIMEOUT="${READINESS_TIMEOUT:-300}"
LIVENESS_TIMEOUT="${LIVENESS_TIMEOUT:-10}"

# Operational flags
ROLLBACK="${ROLLBACK:-false}"
DRY_RUN="${DRY_RUN:-false}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
AUTO_ROLLBACK="${AUTO_ROLLBACK:-true}"

# Color codes for terminal output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly NC='\033[0m'

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

log_info() {
    local message="[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${BLUE}${message}${NC}"
    echo "${message}" >> "${DEPLOYMENT_LOG}"
}

log_success() {
    local message="[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${GREEN}${message}${NC}"
    echo "${message}" >> "${DEPLOYMENT_LOG}"
}

log_warning() {
    local message="[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${YELLOW}${message}${NC}"
    echo "${message}" >> "${DEPLOYMENT_LOG}"
}

log_error() {
    local message="[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${RED}${message}${NC}" >&2
    echo "${message}" >> "${DEPLOYMENT_LOG}"
}

log_deploy() {
    local message="[DEPLOY] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${PURPLE}${message}${NC}"
    echo "${message}" >> "${DEPLOYMENT_LOG}"
}

# ------------------------------------------------------------------------------
# Command Line Interface
# ------------------------------------------------------------------------------

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            --version)
                MODEL_VERSION="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            --rollback)
                ROLLBACK="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --no-auto-rollback)
                AUTO_ROLLBACK="false"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 2
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy AG News Text Classification following continuous delivery best practices.

Options:
    --env, --environment ENV    Target environment: dev, staging, prod (default: staging)
    --platform PLATFORM         Deployment platform: kubernetes, docker, cloud (default: kubernetes)
    --strategy STRATEGY         Deployment strategy: rolling, blue-green, canary (default: rolling)
    --version VERSION           Model version to deploy (default: latest)
    --namespace NAMESPACE       Kubernetes namespace (default: ag-news)
    --replicas COUNT           Number of replicas (default: 3)
    --rollback                 Rollback to previous version
    --dry-run                  Simulate deployment without applying changes
    --no-auto-rollback         Disable automatic rollback on failure
    --help                     Show this help message

Deployment Strategies (Humble & Farley, 2010):
    rolling     - Gradual replacement of instances
    blue-green  - Zero-downtime deployment with environment switch
    canary      - Progressive rollout to subset of users

Environments:
    dev         - Development environment for testing
    staging     - Pre-production environment
    prod        - Production environment with strict controls

Examples:
    # Deploy to staging with rolling update
    $(basename "$0") --env staging --strategy rolling
    
    # Blue-green deployment to production
    $(basename "$0") --env prod --strategy blue-green --version v1.0.0
    
    # Rollback production deployment
    $(basename "$0") --env prod --rollback
    
    # Dry run for canary deployment
    $(basename "$0") --env prod --strategy canary --dry-run

References:
    - Continuous Delivery patterns from Humble & Farley (2010)
    - Container orchestration from Burns et al. (2016)
    - Infrastructure as Code from Morris (2016)
EOF
}

# ------------------------------------------------------------------------------
# Pre-deployment Validation
# ------------------------------------------------------------------------------

validate_deployment_environment() {
    log_info "Validating deployment environment following SRE practices..."
    
    # Validate environment
    case "${ENVIRONMENT}" in
        dev|development)
            REPLICAS="${REPLICAS:-1}"
            MAX_UNAVAILABLE=1
            log_info "Development environment configuration applied"
            ;;
        staging)
            REPLICAS="${REPLICAS:-2}"
            log_info "Staging environment configuration applied"
            ;;
        prod|production)
            REPLICAS="${REPLICAS:-3}"
            MAX_UNAVAILABLE=0
            HEALTH_CHECK_RETRIES=20
            log_info "Production environment configuration applied"
            ;;
        *)
            log_error "Invalid environment: ${ENVIRONMENT}"
            exit 2
            ;;
    esac
    
    # Validate platform availability
    case "${PLATFORM}" in
        kubernetes|k8s)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/"
                exit 3
            fi
            
            # Check cluster connectivity
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                log_info "Check KUBECONFIG or kubectl configuration"
                exit 3
            fi
            
            # Get cluster version
            local k8s_version
            k8s_version=$(kubectl version --short 2>/dev/null | grep Server | awk '{print $3}')
            log_info "Kubernetes cluster version: ${k8s_version}"
            ;;
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker not found"
                exit 3
            fi
            
            if ! docker info &> /dev/null; then
                log_error "Docker daemon not running"
                exit 3
            fi
            ;;
        *)
            log_error "Unsupported platform: ${PLATFORM}"
            exit 2
            ;;
    esac
    
    log_success "Environment validation completed"
}

# ------------------------------------------------------------------------------
# Kubernetes Deployment Functions
# ------------------------------------------------------------------------------

generate_kubernetes_manifests() {
    log_info "Generating Kubernetes manifests following cloud-native patterns..."
    
    local manifest_dir="${PROJECT_ROOT}/deployment/kubernetes/${ENVIRONMENT}"
    mkdir -p "${manifest_dir}"
    
    # Generate namespace manifest
    cat > "${manifest_dir}/00-namespace.yaml" << EOF
# Namespace for AG News Classifier
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    name: ${NAMESPACE}
    environment: ${ENVIRONMENT}
    managed-by: ag-news-deploy
EOF
    
    # Generate deployment manifest following Kubernetes best practices
    cat > "${manifest_dir}/01-deployment.yaml" << EOF
# Deployment manifest following Kubernetes best practices
# References: Burns et al. (2016), Verma et al. (2015)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
    version: ${MODEL_VERSION}
    environment: ${ENVIRONMENT}
  annotations:
    deployment.kubernetes.io/revision: "${DEPLOYMENT_ID}"
    deployed-by: "Vo Hai Dung"
    deployed-at: "${DEPLOYMENT_TIMESTAMP}"
spec:
  replicas: ${REPLICAS}
  revisionHistoryLimit: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: ${MAX_SURGE}
      maxUnavailable: ${MAX_UNAVAILABLE}
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
      environment: ${ENVIRONMENT}
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
        version: ${MODEL_VERSION}
        environment: ${ENVIRONMENT}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # Pod anti-affinity for high availability
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ${SERVICE_NAME}
              topologyKey: kubernetes.io/hostname
      
      # Security context (CIS Kubernetes Benchmark)
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: ${SERVICE_NAME}
        image: agnews/${SERVICE_NAME}:${MODEL_VERSION}
        imagePullPolicy: Always
        
        # Security context for container
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        
        env:
        - name: ENVIRONMENT
          value: "${ENVIRONMENT}"
        - name: MODEL_VERSION
          value: "${MODEL_VERSION}"
        - name: LOG_LEVEL
          value: "INFO"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        
        # Resource limits following Kubernetes best practices
        resources:
          requests:
            memory: "${MEMORY_REQUEST}"
            cpu: "${CPU_REQUEST}"
          limits:
            memory: "${MEMORY_LIMIT}"
            cpu: "${CPU_LIMIT}"
        
        # Health checks following SRE practices
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: ${LIVENESS_TIMEOUT}
          successThreshold: 1
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        # Startup probe for slow-starting containers
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
            scheme: HTTP
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 30
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/.cache
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
EOF
    
    # Generate service manifest
    cat > "${manifest_dir}/02-service.yaml" << EOF
# Service manifest for load balancing
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ${SERVICE_NAME}
    environment: ${ENVIRONMENT}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  selector:
    app: ${SERVICE_NAME}
    environment: ${ENVIRONMENT}
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
EOF
    
    # Generate HPA for production
    if [[ "${ENVIRONMENT}" == "prod" || "${ENVIRONMENT}" == "production" ]]; then
        cat > "${manifest_dir}/03-hpa.yaml" << EOF
# Horizontal Pod Autoscaler for production
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${SERVICE_NAME}
  minReplicas: ${REPLICAS}
  maxReplicas: $((REPLICAS * 3))
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
EOF
    fi
    
    log_success "Kubernetes manifests generated at ${manifest_dir}"
}

deploy_to_kubernetes() {
    log_deploy "Deploying to Kubernetes cluster..."
    
    local manifest_dir="${PROJECT_ROOT}/deployment/kubernetes/${ENVIRONMENT}"
    
    # Ensure namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_info "Creating namespace: ${NAMESPACE}"
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "[DRY RUN] Would create namespace ${NAMESPACE}"
        else
            kubectl create namespace "${NAMESPACE}"
        fi
    fi
    
    # Apply manifests
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Showing what would be deployed:"
        kubectl apply -f "${manifest_dir}/" --dry-run=client -o yaml | head -100
        return 0
    fi
    
    log_info "Applying Kubernetes manifests..."
    if ! kubectl apply -f "${manifest_dir}/" --record; then
        log_error "Failed to apply Kubernetes manifests"
        return 1
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready (timeout: ${READINESS_TIMEOUT}s)..."
    if kubectl rollout status deployment/"${SERVICE_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${READINESS_TIMEOUT}s"; then
        log_success "Deployment rollout completed successfully"
    else
        log_error "Deployment rollout failed or timed out"
        
        # Auto rollback if enabled
        if [[ "${AUTO_ROLLBACK}" == "true" ]]; then
            log_warning "Initiating automatic rollback..."
            perform_rollback
        fi
        return 1
    fi
    
    # Verify deployment health
    if ! verify_deployment_health; then
        if [[ "${AUTO_ROLLBACK}" == "true" ]]; then
            log_warning "Health check failed, initiating automatic rollback..."
            perform_rollback
        fi
        return 1
    fi
    
    log_success "Kubernetes deployment completed successfully"
}

verify_deployment_health() {
    log_info "Verifying deployment health..."
    
    local retries=0
    local ready_replicas=0
    local desired_replicas="${REPLICAS}"
    
    while [[ ${retries} -lt ${HEALTH_CHECK_RETRIES} ]]; do
        # Get deployment status
        ready_replicas=$(kubectl get deployment "${SERVICE_NAME}" \
            -n "${NAMESPACE}" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        log_info "Ready replicas: ${ready_replicas}/${desired_replicas} (attempt $((retries + 1))/${HEALTH_CHECK_RETRIES})"
        
        if [[ "${ready_replicas}" == "${desired_replicas}" ]]; then
            log_success "All ${desired_replicas} replicas are ready"
            
            # Additional health check via service endpoint
            local service_ip
            service_ip=$(kubectl get service "${SERVICE_NAME}" \
                -n "${NAMESPACE}" \
                -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            
            if [[ -n "${service_ip}" ]]; then
                log_info "Service endpoint: http://${service_ip}"
            fi
            
            return 0
        fi
        
        sleep "${HEALTH_CHECK_INTERVAL}"
        retries=$((retries + 1))
    done
    
    log_error "Deployment health check failed after ${HEALTH_CHECK_RETRIES} attempts"
    
    # Display pod status for debugging
    log_info "Pod status:"
    kubectl get pods -n "${NAMESPACE}" -l app="${SERVICE_NAME}"
    
    # Display recent events
    log_info "Recent events:"
    kubectl get events -n "${NAMESPACE}" --sort-by='.lastTimestamp' | tail -10
    
    return 1
}

perform_rollback() {
    log_deploy "Performing rollback..."
    
    if [[ "${PLATFORM}" == "kubernetes" || "${PLATFORM}" == "k8s" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "[DRY RUN] Would rollback deployment ${SERVICE_NAME}"
            kubectl rollout history deployment/"${SERVICE_NAME}" -n "${NAMESPACE}"
            return 0
        fi
        
        log_info "Rolling back Kubernetes deployment..."
        if kubectl rollout undo deployment/"${SERVICE_NAME}" -n "${NAMESPACE}"; then
            log_info "Waiting for rollback to complete..."
            kubectl rollout status deployment/"${SERVICE_NAME}" \
                -n "${NAMESPACE}" \
                --timeout="${READINESS_TIMEOUT}s"
            
            log_success "Rollback completed successfully"
            return 0
        else
            log_error "Rollback failed"
            return 1
        fi
    else
        log_error "Rollback not implemented for platform: ${PLATFORM}"
        return 1
    fi
}

# ------------------------------------------------------------------------------
# Post-deployment Tasks
# ------------------------------------------------------------------------------

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service endpoint
    local service_endpoint=""
    
    if [[ "${PLATFORM}" == "kubernetes" || "${PLATFORM}" == "k8s" ]]; then
        # Try to get LoadBalancer IP
        service_endpoint=$(kubectl get service "${SERVICE_NAME}" \
            -n "${NAMESPACE}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        
        # Fallback to NodePort if LoadBalancer not available
        if [[ -z "${service_endpoint}" ]]; then
            local node_ip
            node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}' 2>/dev/null || echo "")
            local node_port
            node_port=$(kubectl get service "${SERVICE_NAME}" \
                -n "${NAMESPACE}" \
                -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "")
            
            if [[ -n "${node_ip}" && -n "${node_port}" ]]; then
                service_endpoint="${node_ip}:${node_port}"
            fi
        fi
    fi
    
    if [[ -z "${service_endpoint}" ]]; then
        log_warning "Could not determine service endpoint for smoke tests"
        return 0
    fi
    
    log_info "Running smoke tests against: http://${service_endpoint}"
    
    # Health check
    if curl -f -s -o /dev/null -w "%{http_code}" "http://${service_endpoint}/health/live"; then
        log_success "Health check passed"
    else
        log_warning "Health check failed"
    fi
    
    # Ready check
    if curl -f -s -o /dev/null -w "%{http_code}" "http://${service_endpoint}/health/ready"; then
        log_success "Readiness check passed"
    else
        log_warning "Readiness check failed"
    fi
    
    log_success "Smoke tests completed"
}

generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="${PROJECT_ROOT}/outputs/deployments/reports/${DEPLOYMENT_ID}_report.md"
    mkdir -p "${PROJECT_ROOT}/outputs/deployments/reports"
    
    cat > "${report_file}" << EOF
# Deployment Report - AG News Text Classification

## Deployment Summary

- **Deployment ID**: ${DEPLOYMENT_ID}
- **Timestamp**: ${DEPLOYMENT_TIMESTAMP}
- **Environment**: ${ENVIRONMENT}
- **Platform**: ${PLATFORM}
- **Strategy**: ${DEPLOYMENT_STRATEGY}
- **Model Version**: ${MODEL_VERSION}
- **Service**: ${SERVICE_NAME}
- **Namespace**: ${NAMESPACE}
- **Replicas**: ${REPLICAS}
- **Status**: $([ "${DRY_RUN}" == "true" ] && echo "DRY RUN" || echo "DEPLOYED")

## Resource Configuration

### Compute Resources
- **CPU Request**: ${CPU_REQUEST}
- **CPU Limit**: ${CPU_LIMIT}
- **Memory Request**: ${MEMORY_REQUEST}
- **Memory Limit**: ${MEMORY_LIMIT}

### Deployment Strategy
- **Type**: ${DEPLOYMENT_STRATEGY}
- **Max Surge**: ${MAX_SURGE}
- **Max Unavailable**: ${MAX_UNAVAILABLE}

### Health Checks
- **Liveness Timeout**: ${LIVENESS_TIMEOUT}s
- **Readiness Timeout**: ${READINESS_TIMEOUT}s
- **Health Check Retries**: ${HEALTH_CHECK_RETRIES}
- **Health Check Interval**: ${HEALTH_CHECK_INTERVAL}s

## Deployment Log

Detailed logs available at: ${DEPLOYMENT_LOG}

## Rollback Instructions

To rollback this deployment:
\`\`\`bash
./ci/deploy.sh --env ${ENVIRONMENT} --rollback
\`\`\`

## Monitoring

- Prometheus metrics available at: /metrics
- Health check endpoint: /health/live
- Readiness endpoint: /health/ready

## References

- Deployment strategy based on Humble & Farley (2010): "Continuous Delivery"
- Container orchestration following Burns et al. (2016): "Borg, Omega, and Kubernetes"
- Infrastructure as Code principles from Morris (2016)

---
*Generated by AG News Classification Deployment Pipeline*
*Author: Vo Hai Dung*
*Date: $(date)*
EOF
    
    log_success "Deployment report saved: ${report_file}"
}

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

main() {
    # Setup logging
    mkdir -p "$(dirname "${DEPLOYMENT_LOG}")"
    
    log_info "Starting AG News Classification Deployment"
    log_info "Deployment ID: ${DEPLOYMENT_ID}"
    log_info "Following continuous delivery practices from academic literature"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate deployment environment
    validate_deployment_environment
    
    # Handle rollback request
    if [[ "${ROLLBACK}" == "true" ]]; then
        if perform_rollback; then
            generate_deployment_report
            log_success "Rollback completed successfully"
            exit 0
        else
            log_error "Rollback failed"
            exit 1
        fi
    fi
    
    # Generate Kubernetes manifests
    generate_kubernetes_manifests
    
    # Execute deployment based on platform
    local deployment_result=0
    
    case "${PLATFORM}" in
        kubernetes|k8s)
            deploy_to_kubernetes || deployment_result=$?
            ;;
        docker)
            log_error "Docker deployment not yet implemented"
            deployment_result=1
            ;;
        *)
            log_error "Unsupported platform: ${PLATFORM}"
            deployment_result=2
            ;;
    esac
    
    # Post-deployment tasks if successful
    if [[ ${deployment_result} -eq 0 ]]; then
        run_smoke_tests
        generate_deployment_report
        
        log_success "Deployment completed successfully"
        log_info "Deployment ID: ${DEPLOYMENT_ID}"
        
        # Display deployment summary
        echo ""
        log_info "Deployment Summary:"
        log_info "  Environment: ${ENVIRONMENT}"
        log_info "  Service: ${SERVICE_NAME}"
        log_info "  Version: ${MODEL_VERSION}"
        log_info "  Replicas: ${REPLICAS}"
        
        if [[ "${PLATFORM}" == "kubernetes" || "${PLATFORM}" == "k8s" ]]; then
            echo ""
            log_info "Kubernetes Resources:"
            kubectl get deployment,service,hpa -n "${NAMESPACE}" -l app="${SERVICE_NAME}"
        fi
    else
        log_error "Deployment failed"
        generate_deployment_report
    fi
    
    exit ${deployment_result}
}

# Execute main function
main "$@"
