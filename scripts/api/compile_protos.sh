#!/bin/bash
# ================================================================================
# Compile Protocol Buffer Files for gRPC Services
# ================================================================================
# This script compiles all .proto files to generate Python code for gRPC services.
# It handles both message definitions and service stubs generation.
#
# Usage: ./compile_protos.sh [options]
# Options:
#   -c, --clean     Clean generated files before compilation
#   -v, --verbose   Enable verbose output
#   -h, --help      Show help message
#
# Author: Võ Hải Dũng
# License: MIT
# ================================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROTO_DIR="${PROJECT_ROOT}/src/api/grpc/protos"
OUTPUT_DIR="${PROJECT_ROOT}/src/api/grpc/compiled"
PYTHON_OUT="${OUTPUT_DIR}"

# Parse arguments
CLEAN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -c, --clean     Clean generated files before compilation"
            echo "  -v, --verbose   Enable verbose output"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check dependencies
check_dependencies() {
    print_message "$YELLOW" "Checking dependencies..."
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        print_message "$RED" "Error: Python 3 is not installed"
        exit 1
    fi
    
    # Check for protoc
    if ! command -v protoc &> /dev/null; then
        print_message "$RED" "Error: protoc is not installed"
        echo "Please install protobuf compiler:"
        echo "  Ubuntu/Debian: sudo apt-get install protobuf-compiler"
        echo "  macOS: brew install protobuf"
        exit 1
    fi
    
    # Check for grpc tools
    if ! python3 -c "import grpc_tools" 2>/dev/null; then
        print_message "$YELLOW" "Installing grpcio-tools..."
        pip install grpcio-tools
    fi
    
    print_message "$GREEN" "All dependencies satisfied"
}

# Function to clean generated files
clean_generated_files() {
    print_message "$YELLOW" "Cleaning generated files..."
    
    # Remove all *_pb2.py and *_pb2_grpc.py files
    find "${OUTPUT_DIR}" -name "*_pb2.py" -type f -delete 2>/dev/null || true
    find "${OUTPUT_DIR}" -name "*_pb2_grpc.py" -type f -delete 2>/dev/null || true
    find "${OUTPUT_DIR}" -name "*.pyc" -type f -delete 2>/dev/null || true
    find "${OUTPUT_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_message "$GREEN" "Cleaned generated files"
}

# Function to create output directories
create_directories() {
    print_message "$YELLOW" "Creating output directories..."
    
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/common"
    
    # Create __init__.py files if they don't exist
    touch "${OUTPUT_DIR}/__init__.py"
    touch "${OUTPUT_DIR}/common/__init__.py"
    
    print_message "$GREEN" "Directories created"
}

# Function to compile a single proto file
compile_proto() {
    local proto_file=$1
    local proto_name=$(basename "${proto_file}" .proto)
    
    if [ "$VERBOSE" = true ]; then
        print_message "$YELLOW" "Compiling ${proto_name}.proto..."
    fi
    
    # Compile with protoc
    python3 -m grpc_tools.protoc \
        -I"${PROTO_DIR}" \
        --python_out="${PYTHON_OUT}" \
        --grpc_python_out="${PYTHON_OUT}" \
        "${proto_file}"
    
    # Fix imports in generated files
    local pb2_file="${OUTPUT_DIR}/${proto_name}_pb2.py"
    local pb2_grpc_file="${OUTPUT_DIR}/${proto_name}_pb2_grpc.py"
    
    # Fix relative imports for common types
    if [ -f "${pb2_file}" ]; then
        sed -i.bak 's/^import common\./from . import common./' "${pb2_file}" 2>/dev/null || \
        sed -i '' 's/^import common\./from . import common./' "${pb2_file}" 2>/dev/null || true
        rm -f "${pb2_file}.bak"
    fi
    
    if [ -f "${pb2_grpc_file}" ]; then
        sed -i.bak "s/^import ${proto_name}_pb2/from . import ${proto_name}_pb2/" "${pb2_grpc_file}" 2>/dev/null || \
        sed -i '' "s/^import ${proto_name}_pb2/from . import ${proto_name}_pb2/" "${pb2_grpc_file}" 2>/dev/null || true
        rm -f "${pb2_grpc_file}.bak"
    fi
    
    if [ "$VERBOSE" = true ]; then
        print_message "$GREEN" "Compiled ${proto_name}.proto"
    fi
}

# Main compilation function
compile_all_protos() {
    print_message "$YELLOW" "Starting proto compilation..."
    
    # Compile common protos first
    if [ -d "${PROTO_DIR}/common" ]; then
        for proto_file in "${PROTO_DIR}"/common/*.proto; do
            if [ -f "$proto_file" ]; then
                compile_proto "$proto_file"
            fi
        done
    fi
    
    # Compile service protos
    for proto_file in "${PROTO_DIR}"/*.proto; do
        if [ -f "$proto_file" ]; then
            compile_proto "$proto_file"
        fi
    done
    
    print_message "$GREEN" "All proto files compiled successfully"
}

# Function to generate documentation
generate_documentation() {
    if [ "$VERBOSE" = true ]; then
        print_message "$YELLOW" "Generating proto documentation..."
    fi
    
    # Create documentation file
    DOC_FILE="${PROJECT_ROOT}/docs/api_reference/grpc_services.md"
    mkdir -p "$(dirname "${DOC_FILE}")"
    
    cat > "${DOC_FILE}" << EOF
# gRPC Services Documentation

Generated on: $(date)

## Available Services

EOF
    
    # List all proto files
    for proto_file in "${PROTO_DIR}"/*.proto; do
        if [ -f "$proto_file" ]; then
            proto_name=$(basename "${proto_file}" .proto)
            echo "- ${proto_name}" >> "${DOC_FILE}"
        fi
    done
    
    if [ "$VERBOSE" = true ]; then
        print_message "$GREEN" "Documentation generated"
    fi
}

# Function to run tests
run_tests() {
    if [ "$VERBOSE" = true ]; then
        print_message "$YELLOW" "Running import tests..."
    fi
    
    # Test if generated files can be imported
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
try:
    from src.api.grpc.compiled import classification_pb2
    from src.api.grpc.compiled import classification_pb2_grpc
    print('Import test passed')
except ImportError as e:
    print(f'Import test failed: {e}')
    sys.exit(1)
" || {
        print_message "$RED" "Import test failed"
        exit 1
    }
    
    if [ "$VERBOSE" = true ]; then
        print_message "$GREEN" "Tests passed"
    fi
}

# Main execution
main() {
    print_message "$GREEN" "=== Proto Compilation Script ==="
    
    # Check dependencies
    check_dependencies
    
    # Clean if requested
    if [ "$CLEAN" = true ]; then
        clean_generated_files
    fi
    
    # Create directories
    create_directories
    
    # Compile protos
    compile_all_protos
    
    # Generate documentation
    generate_documentation
    
    # Run tests
    run_tests
    
    print_message "$GREEN" "=== Compilation completed successfully ==="
}

# Run main function
main
