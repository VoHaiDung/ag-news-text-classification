"""
Test Suite for AG News Text Classification
===========================================

This test suite provides comprehensive validation for the AG News text
classification system, following established software testing standards
and academic best practices.

Testing Framework:
-----------------
The suite is built on pytest, leveraging its powerful features for:
- Fixture-based test setup (dependency injection)
- Parametrized testing for comprehensive coverage
- Marker-based test organization
- Detailed assertion introspection

Standards Compliance:
--------------------
- IEEE 829-2008: Software and System Test Documentation
- ISO/IEC/IEEE 29119: Software and System Testing
- IEEE 1012-2016: System, Software, and Hardware Verification and Validation

Test Organization:
-----------------
tests/
├── unit/           # Isolated component testing
├── integration/    # Component interaction testing
├── performance/    # Efficiency and scalability testing
└── fixtures/       # Shared test data and utilities

Testing Philosophy:
------------------
Following the Test Pyramid strategy (Cohn, 2009):
- 70% Unit tests: Fast, isolated, numerous
- 20% Integration tests: Component interactions
- 10% End-to-end tests: Full system validation

Quality Metrics:
---------------
- Code Coverage: Target > 90% (statement coverage)
- Branch Coverage: Target > 85% (decision coverage)
- Mutation Score: Target > 75% (fault detection capability)

Author: Võ Hải Dũng
Institution: Academic Research Laboratory
License: MIT
Version: 1.0.0
Date: 2024
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__license__ = "MIT"
__email__ = "research@academic.edu"

# Test suite configuration
TEST_CONFIG = {
    'coverage_threshold': 0.9,
    'performance_baseline': 1.0,  # seconds
    'memory_limit': 8.0,  # GB
    'timeout_default': 300,  # seconds
}

# Export only necessary items to avoid namespace pollution
__all__ = [
    '__version__',
    '__author__',
    '__license__',
    'TEST_CONFIG'
]

# Note: This file intentionally avoids importing from src to prevent
# circular dependencies and ensure test isolation. All necessary
# imports are handled at the test module level or through fixtures
# in conftest.py
