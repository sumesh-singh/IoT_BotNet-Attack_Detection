#!/bin/bash
# Enhanced IoT BotScan Deployment Script
# Author: Kotiwale Sumesh Singh (160124862043)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="enhanced-iot-botscan"
DOCKER_IMAGE="iot-botscan"
DOCKER_TAG="latest"
NAMESPACE="iot-botscan"

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check kubectl (for Kubernetes deployment)
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl is not installed. Kubernetes deployment will be skipped."
    fi
    
    print_success "Dependencies check completed"
}

build_docker_image() {
    print_info "Building Docker image..."
    
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

deploy_with_docker_compose() {
    print_info "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p data cache logs models
    
    # Start services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "Docker Compose deployment completed"
        print_info "Services are starting up. Check status with: docker-compose ps"
    else
        print_error "Docker Compose deployment failed"
        exit 1
    fi
}

deploy_with_kubernetes() {
    print_info "Deploying with Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not available. Please install kubectl first."
        return 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/deployment.yaml
    
    if [ $? -eq 0 ]; then
        print_success "Kubernetes deployment completed"
        print_info "Check deployment status with: kubectl get pods -n ${NAMESPACE}"
    else
        print_error "Kubernetes deployment failed"
        exit 1
    fi
}

run_tests() {
    print_info "Running tests..."
    
    # Run unit tests
    python -m pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed"
    fi
}

check_health() {
    print_info "Checking application health..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy"
    else
        print_warning "API health check failed"
    fi
    
    # Check WebSocket
    if curl -f http://localhost:8000/ws > /dev/null 2>&1; then
        print_success "WebSocket is accessible"
    else
        print_warning "WebSocket check failed"
    fi
}

show_status() {
    print_info "Application Status:"
    echo "===================="
    
    # Docker Compose status
    if [ -f docker-compose.yml ]; then
        echo "Docker Compose Services:"
        docker-compose ps
        echo ""
    fi
    
    # Kubernetes status
    if command -v kubectl &> /dev/null; then
        echo "Kubernetes Pods:"
        kubectl get pods -n ${NAMESPACE} 2>/dev/null || echo "No Kubernetes deployment found"
        echo ""
    fi
    
    # Application URLs
    echo "Application URLs:"
    echo "  - Dashboard: http://localhost:8000/dashboard"
    echo "  - Analytics: http://localhost:8000/analytics"
    echo "  - GraphQL API: http://localhost:8000/graphql"
    echo "  - WebSocket: ws://localhost:8000/ws"
    echo "  - Health Check: http://localhost:8000/health"
    echo ""
}

cleanup() {
    print_info "Cleaning up..."
    
    # Stop Docker Compose services
    if [ -f docker-compose.yml ]; then
        docker-compose down
    fi
    
    # Remove Kubernetes deployment
    if command -v kubectl &> /dev/null; then
        kubectl delete -f k8s/deployment.yaml 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

show_help() {
    echo "Enhanced IoT BotScan Deployment Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  build          Build Docker image"
    echo "  deploy-docker  Deploy with Docker Compose"
    echo "  deploy-k8s     Deploy with Kubernetes"
    echo "  deploy-all     Deploy with both Docker Compose and Kubernetes"
    echo "  test           Run tests"
    echo "  health         Check application health"
    echo "  status         Show application status"
    echo "  cleanup        Clean up deployments"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build Docker image"
    echo "  $0 deploy-docker            # Deploy with Docker Compose"
    echo "  $0 deploy-k8s              # Deploy with Kubernetes"
    echo "  $0 deploy-all               # Deploy with both methods"
    echo "  $0 status                  # Show status"
    echo "  $0 cleanup                 # Clean up"
    echo ""
}

# Main script
main() {
    case "${1:-help}" in
        "build")
            check_dependencies
            build_docker_image
            ;;
        "deploy-docker")
            check_dependencies
            build_docker_image
            deploy_with_docker_compose
            check_health
            show_status
            ;;
        "deploy-k8s")
            check_dependencies
            build_docker_image
            deploy_with_kubernetes
            show_status
            ;;
        "deploy-all")
            check_dependencies
            build_docker_image
            deploy_with_docker_compose
            deploy_with_kubernetes
            check_health
            show_status
            ;;
        "test")
            run_tests
            ;;
        "health")
            check_health
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
