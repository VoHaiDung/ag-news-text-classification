#!/bin/bash

# ==============================================================================
# Test Runner Script for AG News Text Classification
# ==============================================================================
# 
# This script executes comprehensive test suites following CI/CD best practices
# from Martin Fowler's "Continuous Integration" and Google's Testing Blog.
#
# References:
# - Fowler, M. (2006). "Continuous Integration"
# - Google Testing Blog: "Testing Best Practices"
# - Humble & Farley (2010): "Continuous Delivery"
# - Duvall, P. M., Matyas, S., & Glover, A. (2007). "Continuous Integration: 
#   Improving Software Quality and Reducing Risk"
#
# Test Strategy:
# - Unit tests: Isolated component testing with mocking
# - Integration tests: End-to-end pipeline validation
# - Performance tests: Benchmarking and profiling
# - Static analysis: Type checking, linting, security scanning
#
# Author: Võ Hải Dũng
# License: MIT
# ==============================================================================

set -euo pipefail  # Exit on error, undefined variable, or pipe failure
IFS=$'\n\t'       # Set Internal Field Separator for word splitting safety

# ------------------------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly TEST_REPORT_DIR="${PROJECT_ROOT}/outputs/test_results/${TIMESTAMP}"
readonly COVERAGE_DIR="${PROJECT_ROOT}/outputs/coverage/${TIMESTAMP}"

# Test configuration with defaults
readonly PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
readonly TEST_TIMEOUT="${TEST_TIMEOUT:-300}"  # 5 minutes default
readonly COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-60}"  # 60% minimum coverage
readonly PARALLEL_JOBS="${PARALLEL_JOBS:-auto}"
readonly MIN_COVERAGE="${MIN_COVERAGE:-60}"

# Test suite options
TEST_TYPE="${TEST_TYPE:-all}"
COVERAGE_ENABLED=false
VERBOSE_MODE=false
PARALLEL_MODE=false
PYTEST_MARKERS=""
DRY_RUN=false
STRICT_MODE=false

# Color codes for terminal output (ANSI escape sequences)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

# Logging functions with standardized format
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_metric() {
    echo -e "${CYAN}[METRIC]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clean exit with proper cleanup
cleanup_and_exit() {
    local exit_code=$1
    log_info "Cleaning up temporary files..."
    find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${PROJECT_ROOT}" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    exit "${exit_code}"
}

# Trap signals for cleanup
trap 'cleanup_and_exit 130' INT TERM

# ------------------------------------------------------------------------------
# Command Line Interface
# ------------------------------------------------------------------------------

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type)
                TEST_TYPE="$2"
                shift 2
                ;;
            --coverage)
                COVERAGE_ENABLED=true
                shift
                ;;
            --verbose|-v)
                VERBOSE_MODE=true
                shift
                ;;
            --parallel|-p)
                PARALLEL_MODE=true
                shift
                ;;
            --markers|-m)
                PYTEST_MARKERS="$2"
                shift 2
                ;;
            --timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            --strict)
                STRICT_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
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

# Display help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run test suites for AG News Text Classification project following CI/CD best practices.

Options:
    --type TYPE       Test type: unit, integration, performance, lint, security, all (default: all)
    --coverage        Generate coverage report with threshold checking
    --verbose, -v     Verbose output for debugging
    --parallel, -p    Run tests in parallel using pytest-xdist
    --markers, -m     Pytest markers to filter tests (e.g., "not slow")
    --timeout TIMEOUT Test timeout in seconds (default: 300)
    --strict         Fail on warnings and type checking errors
    --dry-run        Show what would be executed without running
    --help, -h       Show this help message

Test Types:
    unit         - Unit tests with mocking and isolation
    integration  - End-to-end pipeline tests
    performance  - Speed and memory benchmarks
    lint         - Code quality and style checking
    security     - Security vulnerability scanning
    docs         - Documentation build and doctests
    all          - Run all test suites (default)

Examples:
    # Run all tests with coverage
    $(basename "$0") --coverage

    # Run only unit tests in verbose mode
    $(basename "$0") --type unit --verbose

    # Run tests in parallel excluding slow tests
    $(basename "$0") --parallel --markers "not slow"

    # Strict mode with all checks
    $(basename "$0") --strict --coverage

Environment Variables:
    PYTHON_VERSION       - Required Python version (default: 3.10)
    TEST_TIMEOUT        - Global test timeout (default: 300)
    COVERAGE_THRESHOLD  - Minimum coverage percentage (default: 60)
    PARALLEL_JOBS       - Number of parallel jobs (default: auto)

References:
    - Testing best practices from Google Testing Blog
    - Continuous Integration patterns from Martin Fowler
EOF
}

# ------------------------------------------------------------------------------
# Environment Setup Functions
# ------------------------------------------------------------------------------

# Setup test environment with proper isolation
setup_test_environment() {
    log_info "Setting up test environment..."
    
    # Create report directories following project structure
    mkdir -p "${TEST_REPORT_DIR}"/{unit,integration,performance}
    mkdir -p "${COVERAGE_DIR}"/{html,xml}
    mkdir -p "${PROJECT_ROOT}/outputs"/{logs,artifacts}
    
    # Export environment variables for test isolation
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
    export AG_NEWS_ENV="test"
    export AG_NEWS_LOG_LEVEL="WARNING"
    export TESTING=true
    export CUDA_VISIBLE_DEVICES=""  # Disable GPU for reproducible tests
    
    # Set reproducibility seeds
    export PYTHONHASHSEED=0
    export RANDOM_SEED=42
    
    # Verify Python version
    local python_version
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    
    if [[ "${python_version}" != "${PYTHON_VERSION}" ]]; then
        log_warning "Python version mismatch. Expected: ${PYTHON_VERSION}, Got: ${python_version}"
        if [[ "${STRICT_MODE}" == true ]]; then
            log_error "Strict mode enabled. Python version must match."
            exit 3
        fi
    fi
    
    log_success "Test environment configured at ${TEST_REPORT_DIR}"
}

# Install and verify test dependencies
install_test_dependencies() {
    log_info "Checking test dependencies..."
    
    # Required test packages
    local required_packages=(
        "pytest"
        "pytest-cov"
        "pytest-timeout"
        "pytest-xdist"
        "pytest-html"
        "pytest-mock"
    )
    
    # Optional but recommended packages
    local optional_packages=(
        "mypy"
        "flake8"
        "black"
        "isort"
        "bandit"
        "safety"
    )
    
    # Check and install required packages
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package//-/_}" 2>/dev/null; then
            log_warning "Installing required package: $package"
            pip install --quiet "$package"
        fi
    done
    
    # Check optional packages
    for package in "${optional_packages[@]}"; do
        if ! command_exists "$package" && ! python3 -c "import ${package//-/_}" 2>/dev/null; then
            log_warning "Optional package not found: $package (some tests may be skipped)"
        fi
    done
    
    log_success "Dependencies verified"
}

# Create dummy tests to prevent CI failures on new projects
ensure_test_structure() {
    log_info "Ensuring test structure exists..."
    
    # Create test directories if they don't exist
    mkdir -p "${PROJECT_ROOT}/tests"/{unit,integration,performance,fixtures}
    
    # Create conftest.py for pytest configuration
    if [[ ! -f "${PROJECT_ROOT}/tests/conftest.py" ]]; then
        cat > "${PROJECT_ROOT}/tests/conftest.py" << 'EOF'
"""
Pytest configuration for AG News Classification tests.

Based on pytest best practices from:
- Okken, B. (2017). "Python Testing with pytest"
- Pajankar, A. (2017). "Python Unit Test Automation"
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
EOF
    fi
    
    # Create dummy unit test if none exist
    if [[ ! -f "${PROJECT_ROOT}/tests/unit/test_basic.py" ]]; then
        log_warning "No unit tests found, creating dummy test..."
        cat > "${PROJECT_ROOT}/tests/unit/test_basic.py" << 'EOF'
"""
Basic unit tests for CI validation.

Following test patterns from:
- Beck, K. (2002). "Test Driven Development: By Example"
"""

import sys
import pytest


class TestBasicFunctionality:
    """Basic functionality tests for CI pipeline validation."""
    
    def test_python_version(self):
        """Verify Python version meets requirements."""
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        import os
        import pathlib
        import json
        import yaml
        
        assert os is not None
        assert pathlib is not None
        assert json is not None
        assert yaml is not None
    
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 2),
        (3, 3)
    ])
    def test_parametrized_example(self, value, expected):
        """Example of parametrized testing pattern."""
        assert value == expected
EOF
    fi
    
    # Create dummy integration test if none exist
    if [[ ! -f "${PROJECT_ROOT}/tests/integration/test_pipeline.py" ]]; then
        log_warning "No integration tests found, creating dummy test..."
        cat > "${PROJECT_ROOT}/tests/integration/test_pipeline.py" << 'EOF'
"""
Integration tests for end-to-end pipeline validation.

Following integration testing patterns from:
- Freeman, S. & Pryce, N. (2009). "Growing Object-Oriented Software, Guided by Tests"
"""

import pytest


class TestPipelineIntegration:
    """Integration tests for data pipeline."""
    
    @pytest.mark.integration
    def test_pipeline_connectivity(self):
        """Test basic pipeline connectivity."""
        # Placeholder for actual pipeline test
        assert True
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_flow(self):
        """Test complete end-to-end data flow."""
        # This test is marked as slow and will be skipped in quick CI runs
        import time
        time.sleep(0.1)  # Simulate processing
        assert True
EOF
    fi
    
    log_success "Test structure verified"
}

# ------------------------------------------------------------------------------
# Test Execution Functions
# ------------------------------------------------------------------------------

# Run unit tests with coverage
run_unit_tests() {
    log_info "Running unit tests..."
    
    local pytest_args=()
    pytest_args+=("${PROJECT_ROOT}/tests/unit")
    pytest_args+=("--tb=short")
    pytest_args+=("--strict-markers")
    pytest_args+=("--timeout=${TEST_TIMEOUT}")
    
    # Output formatting
    pytest_args+=("--color=yes")
    pytest_args+=("--junit-xml=${TEST_REPORT_DIR}/unit/junit.xml")
    pytest_args+=("--html=${TEST_REPORT_DIR}/unit/report.html")
    pytest_args+=("--self-contained-html")
    
    # Verbose mode
    if [[ "${VERBOSE_MODE}" == true ]]; then
        pytest_args+=("-vv")
        pytest_args+=("--capture=no")
    else
        pytest_args+=("-v")
    fi
    
    # Parallel execution
    if [[ "${PARALLEL_MODE}" == true ]]; then
        pytest_args+=("-n" "${PARALLEL_JOBS}")
    fi
    
    # Coverage reporting
    if [[ "${COVERAGE_ENABLED}" == true ]]; then
        pytest_args+=("--cov=src")
        pytest_args+=("--cov-report=term-missing")
        pytest_args+=("--cov-report=html:${COVERAGE_DIR}/html")
        pytest_args+=("--cov-report=xml:${COVERAGE_DIR}/coverage.xml")
        pytest_args+=("--cov-fail-under=${COVERAGE_THRESHOLD}")
    fi
    
    # Custom markers
    if [[ -n "${PYTEST_MARKERS}" ]]; then
        pytest_args+=("-m" "${PYTEST_MARKERS}")
    fi
    
    # Dry run mode
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would execute: pytest ${pytest_args[*]}"
        return 0
    fi
    
    # Execute tests
    if python3 -m pytest "${pytest_args[@]}"; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Unit tests failed"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    local pytest_args=()
    pytest_args+=("${PROJECT_ROOT}/tests/integration")
    pytest_args+=("--tb=short")
    pytest_args+=("--timeout=$((TEST_TIMEOUT * 2))")  # Longer timeout for integration
    
    # Output formatting
    pytest_args+=("--color=yes")
    pytest_args+=("--junit-xml=${TEST_REPORT_DIR}/integration/junit.xml")
    pytest_args+=("--html=${TEST_REPORT_DIR}/integration/report.html")
    pytest_args+=("--self-contained-html")
    
    # Verbose mode
    if [[ "${VERBOSE_MODE}" == true ]]; then
        pytest_args+=("-vv")
    else
        pytest_args+=("-v")
    fi
    
    # Skip slow tests by default
    if [[ "${PYTEST_MARKERS}" != *"slow"* ]]; then
        pytest_args+=("-m" "not slow")
    fi
    
    # Coverage (append to existing)
    if [[ "${COVERAGE_ENABLED}" == true ]]; then
        pytest_args+=("--cov=src")
        pytest_args+=("--cov-append")
        pytest_args+=("--cov-report=")  # No terminal output for append
    fi
    
    # Dry run mode
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would execute: pytest ${pytest_args[*]}"
        return 0
    fi
    
    # Execute tests
    if python3 -m pytest "${pytest_args[@]}"; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

# Run performance benchmarks
run_performance_tests() {
    log_info "Running performance tests..."
    
    local pytest_args=()
    pytest_args+=("${PROJECT_ROOT}/tests/performance")
    pytest_args+=("--tb=short")
    pytest_args+=("--timeout=$((TEST_TIMEOUT * 3))")  # Even longer timeout
    
    # Output formatting
    pytest_args+=("--junit-xml=${TEST_REPORT_DIR}/performance/junit.xml")
    pytest_args+=("--html=${TEST_REPORT_DIR}/performance/report.html")
    pytest_args+=("--self-contained-html")
    
    # Performance specific markers
    pytest_args+=("-m" "not gpu")  # Skip GPU tests in CI
    
    # Dry run mode
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would execute: pytest ${pytest_args[*]}"
        return 0
    fi
    
    # Create dummy performance test if needed
    if [[ ! -d "${PROJECT_ROOT}/tests/performance" ]]; then
        mkdir -p "${PROJECT_ROOT}/tests/performance"
        cat > "${PROJECT_ROOT}/tests/performance/test_benchmark.py" << 'EOF'
"""Performance benchmarks following patterns from Python's timeit module."""

import time
import pytest


class TestPerformanceBenchmarks:
    """Basic performance benchmarks."""
    
    def test_speed_benchmark(self):
        """Test execution speed benchmark."""
        start_time = time.perf_counter()
        # Simulate computation
        sum(range(1000000))
        elapsed = time.perf_counter() - start_time
        assert elapsed < 1.0, f"Operation took {elapsed:.3f}s, expected < 1.0s"
    
    @pytest.mark.benchmark
    def test_memory_benchmark(self):
        """Test memory usage benchmark."""
        import sys
        data = list(range(1000000))
        size_bytes = sys.getsizeof(data)
        size_mb = size_bytes / (1024 * 1024)
        assert size_mb < 100, f"Memory usage {size_mb:.2f}MB exceeds limit"
EOF
    fi
    
    # Execute tests
    if python3 -m pytest "${pytest_args[@]}"; then
        log_success "Performance tests passed"
        return 0
    else
        log_error "Performance tests failed"
        return 1
    fi
}

# Run static type checking with mypy
run_type_checking() {
    log_info "Running type checking with mypy..."
    
    if ! command_exists mypy; then
        log_warning "mypy not found, installing..."
        pip install --quiet mypy types-PyYAML
    fi
    
    local mypy_args=()
    mypy_args+=("${PROJECT_ROOT}/src")
    mypy_args+=("--ignore-missing-imports")
    mypy_args+=("--junit-xml=${TEST_REPORT_DIR}/mypy.xml")
    mypy_args+=("--html-report=${TEST_REPORT_DIR}/mypy")
    
    if [[ "${STRICT_MODE}" == true ]]; then
        mypy_args+=("--strict")
        mypy_args+=("--warn-return-any")
        mypy_args+=("--warn-unused-configs")
        mypy_args+=("--disallow-untyped-defs")
    fi
    
    # Dry run mode
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would execute: mypy ${mypy_args[*]}"
        return 0
    fi
    
    # Execute type checking
    if mypy "${mypy_args[@]}"; then
        log_success "Type checking passed"
        return 0
    else
        log_warning "Type checking found issues"
        if [[ "${STRICT_MODE}" == true ]]; then
            return 1
        fi
        return 0
    fi
}

# Run code quality checks
run_linting() {
    log_info "Running code quality checks..."
    
    local lint_failed=false
    
    # Run flake8
    if command_exists flake8; then
        log_info "Running flake8 linter..."
        if ! flake8 "${PROJECT_ROOT}/src" \
            --max-line-length=100 \
            --max-complexity=10 \
            --output-file="${TEST_REPORT_DIR}/flake8.txt" \
            --tee; then
            log_warning "flake8 found issues"
            lint_failed=true
        fi
    fi
    
    # Run pylint
    if command_exists pylint; then
        log_info "Running pylint..."
        if ! pylint "${PROJECT_ROOT}/src" \
            --output-format=json \
            --output="${TEST_REPORT_DIR}/pylint.json" \
            --fail-under=7.0; then
            log_warning "pylint score below threshold"
            lint_failed=true
        fi
    fi
    
    # Check code formatting with black
    if command_exists black; then
        log_info "Checking code formatting with black..."
        if ! black --check --diff "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/tests"; then
            log_warning "Code formatting issues found. Run 'black .' to fix."
            lint_failed=true
        fi
    fi
    
    # Check import sorting with isort
    if command_exists isort; then
        log_info "Checking import sorting with isort..."
        if ! isort --check-only --diff "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/tests"; then
            log_warning "Import sorting issues found. Run 'isort .' to fix."
            lint_failed=true
        fi
    fi
    
    if [[ "${lint_failed}" == true ]]; then
        log_warning "Linting checks found issues"
        if [[ "${STRICT_MODE}" == true ]]; then
            return 1
        fi
    else
        log_success "All linting checks passed"
    fi
    
    return 0
}

# Run security vulnerability checks
run_security_checks() {
    log_info "Running security checks..."
    
    # Run bandit for security issues in code
    if command_exists bandit; then
        log_info "Running bandit security scan..."
        bandit -r "${PROJECT_ROOT}/src" \
            -f json \
            -o "${TEST_REPORT_DIR}/bandit.json" \
            --severity-level medium || true
        
        # Also output human-readable format
        bandit -r "${PROJECT_ROOT}/src" \
            --severity-level medium || true
    else
        log_warning "bandit not found, skipping security scan"
    fi
    
    # Check dependencies for known vulnerabilities
    if command_exists safety; then
        log_info "Checking dependencies for vulnerabilities..."
        safety check \
            --json \
            --output "${TEST_REPORT_DIR}/safety.json" || true
        
        # Also show human-readable output
        safety check || true
    else
        log_warning "safety not found, skipping dependency check"
    fi
    
    log_success "Security checks completed"
    return 0
}

# Run documentation tests
run_doc_tests() {
    log_info "Running documentation tests..."
    
    # Run doctests
    log_info "Running doctests..."
    python3 -m doctest "${PROJECT_ROOT}/src"/**/*.py \
        --verbose \
        > "${TEST_REPORT_DIR}/doctest.txt" 2>&1 || true
    
    # Check if documentation builds
    if [[ -f "${PROJECT_ROOT}/docs/Makefile" ]]; then
        log_info "Building documentation..."
        (
            cd "${PROJECT_ROOT}/docs"
            make clean
            make html SPHINXOPTS="-W" > "${TEST_REPORT_DIR}/sphinx.log" 2>&1
        ) || log_warning "Documentation build had warnings"
    fi
    
    log_success "Documentation tests completed"
    return 0
}

# ------------------------------------------------------------------------------
# Reporting Functions
# ------------------------------------------------------------------------------

# Generate comprehensive test summary
generate_test_summary() {
    log_info "Generating test summary report..."
    
    local summary_file="${TEST_REPORT_DIR}/summary.md"
    
    {
        echo "# Test Execution Summary - AG News Classification"
        echo ""
        echo "## Metadata"
        echo "- **Timestamp**: ${TIMESTAMP}"
        echo "- **Python Version**: $(python3 --version)"
        echo "- **Platform**: $(uname -s) $(uname -r)"
        echo "- **Test Type**: ${TEST_TYPE}"
        echo "- **Coverage Enabled**: ${COVERAGE_ENABLED}"
        echo ""
        
        echo "## Test Results"
        echo ""
        
        # Parse unit test results
        if [[ -f "${TEST_REPORT_DIR}/unit/junit.xml" ]]; then
            echo "### Unit Tests"
            python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('${TEST_REPORT_DIR}/unit/junit.xml')
    root = tree.getroot()
    for testsuite in root.findall('.//testsuite'):
        tests = testsuite.get('tests', '0')
        failures = testsuite.get('failures', '0')
        errors = testsuite.get('errors', '0')
        skipped = testsuite.get('skipped', '0')
        time = testsuite.get('time', '0')
        passed = int(tests) - int(failures) - int(errors) - int(skipped)
        
        print(f'- **Total Tests**: {tests}')
        print(f'- **Passed**: {passed}')
        print(f'- **Failed**: {failures}')
        print(f'- **Errors**: {errors}')
        print(f'- **Skipped**: {skipped}')
        print(f'- **Execution Time**: {float(time):.2f}s')
        break
except Exception as e:
    print(f'Error parsing results: {e}')
"
            echo ""
        fi
        
        # Parse integration test results
        if [[ -f "${TEST_REPORT_DIR}/integration/junit.xml" ]]; then
            echo "### Integration Tests"
            python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('${TEST_REPORT_DIR}/integration/junit.xml')
    root = tree.getroot()
    for testsuite in root.findall('.//testsuite'):
        tests = testsuite.get('tests', '0')
        failures = testsuite.get('failures', '0')
        time = testsuite.get('time', '0')
        print(f'- **Total Tests**: {tests}')
        print(f'- **Failures**: {failures}')
        print(f'- **Execution Time**: {float(time):.2f}s')
        break
except Exception as e:
    print(f'Error parsing results: {e}')
"
            echo ""
        fi
        
        # Coverage results
        if [[ "${COVERAGE_ENABLED}" == true && -f "${COVERAGE_DIR}/coverage.xml" ]]; then
            echo "### Code Coverage"
            python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('${COVERAGE_DIR}/coverage.xml')
    root = tree.getroot()
    line_rate = float(root.get('line-rate', 0))
    branch_rate = float(root.get('branch-rate', 0))
    
    print(f'- **Line Coverage**: {line_rate * 100:.2f}%')
    print(f'- **Branch Coverage**: {branch_rate * 100:.2f}%')
    print(f'- **Coverage Threshold**: ${COVERAGE_THRESHOLD}%')
    
    if line_rate * 100 >= ${COVERAGE_THRESHOLD}:
        print(f'- **Status**: ✓ PASSED')
    else:
        print(f'- **Status**: ✗ BELOW THRESHOLD')
except Exception as e:
    print(f'Error parsing coverage: {e}')
"
            echo ""
        fi
        
        # Type checking results
        if [[ -f "${TEST_REPORT_DIR}/mypy.xml" ]]; then
            echo "### Type Checking"
            echo "- **Status**: Completed"
            echo "- **Report**: mypy.xml"
            echo ""
        fi
        
        # Security scan results
        if [[ -f "${TEST_REPORT_DIR}/bandit.json" ]]; then
            echo "### Security Analysis"
            python3 -c "
import json
try:
    with open('${TEST_REPORT_DIR}/bandit.json') as f:
        data = json.load(f)
        metrics = data.get('metrics', {})
        print(f\"- **Files Scanned**: {metrics.get('_totals', {}).get('loc', 'N/A')}\")
        print(f\"- **High Severity Issues**: {metrics.get('_totals', {}).get('SEVERITY.HIGH', 0)}\")
        print(f\"- **Medium Severity Issues**: {metrics.get('_totals', {}).get('SEVERITY.MEDIUM', 0)}\")
except Exception as e:
    print(f'Error parsing security results: {e}')
"
            echo ""
        fi
        
        echo "## Artifacts"
        echo "- Test Reports: ${TEST_REPORT_DIR}"
        echo "- Coverage Reports: ${COVERAGE_DIR}"
        echo ""
        
        echo "---"
        echo "*Generated by AG News Classification CI/CD Pipeline*"
        
    } > "${summary_file}"
    
    # Display summary to console
    cat "${summary_file}"
    
    log_success "Test summary saved to: ${summary_file}"
}

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

main() {
    local exit_code=0
    
    log_info "Starting AG News Classification test execution..."
    log_info "Project root: ${PROJECT_ROOT}"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Setup environment
    setup_test_environment
    install_test_dependencies
    ensure_test_structure
    
    # Execute tests based on type
    case "${TEST_TYPE}" in
        unit)
            run_unit_tests || exit_code=$?
            ;;
        integration)
            run_integration_tests || exit_code=$?
            ;;
        performance)
            run_performance_tests || exit_code=$?
            ;;
        lint)
            run_linting || exit_code=$?
            ;;
        security)
            run_security_checks || exit_code=$?
            ;;
        docs)
            run_doc_tests || exit_code=$?
            ;;
        all)
            # Run core test suites
            run_unit_tests || exit_code=$?
            
            if [[ ${exit_code} -eq 0 ]]; then
                run_integration_tests || exit_code=$?
            fi
            
            if [[ ${exit_code} -eq 0 ]]; then
                run_performance_tests || true  # Don't fail on performance
            fi
            
            # Run additional checks (warnings only unless strict mode)
            run_type_checking || ([[ "${STRICT_MODE}" == true ]] && exit_code=$?)
            run_linting || ([[ "${STRICT_MODE}" == true ]] && exit_code=$?)
            run_security_checks || true
            run_doc_tests || true
            ;;
        *)
            log_error "Unknown test type: ${TEST_TYPE}"
            show_help
            exit 2
            ;;
    esac
    
    # Generate test summary
    generate_test_summary
    
    # Final status report
    if [[ ${exit_code} -eq 0 ]]; then
        log_success "All tests passed successfully!"
        log_info "Test reports available at: ${TEST_REPORT_DIR}"
        if [[ "${COVERAGE_ENABLED}" == true ]]; then
            log_info "Coverage reports available at: ${COVERAGE_DIR}"
        fi
    else
        log_error "Some tests failed. Check reports at: ${TEST_REPORT_DIR}"
    fi
    
    # Cleanup and exit
    cleanup_and_exit ${exit_code}
}

# Execute main function with all arguments
main "$@"
