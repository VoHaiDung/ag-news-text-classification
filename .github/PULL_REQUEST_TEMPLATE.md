<!--
================================================================================
PULL REQUEST TEMPLATE - AG NEWS CLASSIFICATION SYSTEM
================================================================================
This template adheres to software engineering best practices and academic 
standards for code contribution in machine learning research projects.

References:
- IEEE Standard for Software Reviews and Audits (IEEE 1028-2008)
- ACM Guidelines for Scientific Software Development
- Google Engineering Practices Documentation
- Open Source Contribution Guidelines

Please complete all applicable sections to facilitate efficient code review.
Incomplete pull requests may experience delays in the review process.
================================================================================
-->

## 1. PULL REQUEST METADATA

### 1.1 PR Type and Scope
<!-- Format: [TYPE] Brief descriptive title -->
<!-- Types: FEATURE, BUGFIX, ENHANCEMENT, REFACTOR, DOCS, TEST, PERF, BUILD, CI, STYLE -->

**PR Title:** [TYPE] 

### 1.2 Change Classification
<!-- Select the primary category of this change -->
- [ ] **Feature Implementation**: New functionality or capability
- [ ] **Bug Fix**: Correction of defective behavior
- [ ] **Performance Optimization**: Speed or efficiency improvements
- [ ] **Refactoring**: Code restructuring without behavior change
- [ ] **Documentation**: Documentation updates only
- [ ] **Testing**: Test additions or modifications
- [ ] **Build/CI**: Build system or CI/CD pipeline changes
- [ ] **Dependencies**: Dependency updates or modifications
- [ ] **Research Implementation**: Academic algorithm or method implementation

### 1.3 Priority Level
- [ ] **Critical**: Blocking issue or security vulnerability
- [ ] **High**: Important feature or significant bug fix
- [ ] **Medium**: Standard enhancement or non-critical fix
- [ ] **Low**: Minor improvement or cosmetic change

## 2. CHANGE DESCRIPTION

### 2.1 Executive Summary
<!-- Provide a comprehensive description of the changes in this PR -->
```
Summary of changes:
```

### 2.2 Motivation and Context
<!-- Explain the rationale behind these changes -->
```
Motivation:
- Problem being solved:
- Why this approach was chosen:
- Expected benefits:
```

### 2.3 Related Issues and Dependencies
<!-- Link related issues, PRs, and dependencies using GitHub keywords -->
```
Issue Resolution:
- Closes #
- Fixes #
- Resolves #

Related Work:
- Related to #
- Depends on #
- Blocks #
```
## 3. TECHNICAL IMPLEMENTATION

### 3.1 Architectural Changes
<!-- Describe any architectural modifications -->
```
Architecture Impact:
- Modified components:
- New components added:
- Components removed:
- Interface changes:
```

### 3.2 Implementation Details
<!-- Provide technical details of the implementation -->
```python
# Key implementation aspects
"""
1. Algorithm/Method:
   - Name:
   - Time Complexity: O()
   - Space Complexity: O()
   
2. Design Patterns Used:
   - Pattern name and justification
   
3. Data Structures:
   - New structures introduced
   - Modifications to existing structures
"""
```

### 3.3 File Changes Summary
<!-- List all modified files with brief descriptions -->
| File Path | Change Type | Description |
|-----------|-------------|-------------|
| `src/models/transformers/deberta/deberta_v3.py` | Modified | Added gradient checkpointing support |
| `configs/training/standard/base_training.yaml` | Modified | Updated default hyperparameters |
| `tests/unit/models/test_deberta.py` | Added | Unit tests for new functionality |
| | | |

### 3.4 Code Modifications Breakdown
```diff
# Summary of additions and deletions
+ Lines added: 
- Lines removed: 
~ Lines modified: 
```

## 4. TESTING AND VALIDATION

### 4.1 Test Coverage
<!-- Describe the testing approach and coverage -->
```yaml
Testing Strategy:
  Unit Tests:
    - Added: # Number of new unit tests
    - Modified: # Number of modified tests
    - Coverage: # Percentage coverage
    
  Integration Tests:
    - Added: # Number of new integration tests
    - Modified: # Number of modified tests
    - Scenarios Covered: # List key scenarios
    
  End-to-End Tests:
    - Added: # Number of E2E tests
    - Test Environments: # List environments tested
```

### 4.2 Test Execution Results
<!-- Provide test execution output -->
```bash
# Test execution command and results
pytest tests/ -v --cov=src --cov-report=term-missing

# Output:
# ================= test session starts =================
# collected X items
# 
# tests/unit/... PASSED [100%]
# 
# ---------- coverage: platform linux, python 3.10.x -----------
# Name                     Stmts   Miss  Cover   Missing
# ------------------------------------------------------
# src/module/file.py         100      5    95%   12-16
# ------------------------------------------------------
# TOTAL                      XXXX    XX    XX%
# 
# ================= X passed in X.XXs =================
```

### 4.3 Validation Methodology
<!-- Describe how the changes were validated -->
```yaml
Validation Approach:
  - Dataset Used: # e.g., AG News test set
  - Validation Metrics: # e.g., accuracy, F1, precision, recall
  - Baseline Comparison: # Results vs. baseline
  - Statistical Significance: # p-value if applicable
```

## 5. PERFORMANCE ANALYSIS

### 5.1 Performance Metrics
<!-- Provide quantitative performance comparisons -->
| Metric | Baseline | This PR | Change | Relative Change |
|--------|----------|---------|--------|-----------------|
| Training Speed (samples/sec) | | | | % |
| Inference Latency (ms) | | | | % |
| Memory Usage (GB) | | | | % |
| Model Accuracy (%) | | | | % |
| F1 Score | | | | % |

### 5.2 Computational Complexity Analysis
```yaml
Algorithm Complexity:
  Time Complexity:
    - Before: O()
    - After: O()
    - Justification: 
    
  Space Complexity:
    - Before: O()
    - After: O()
    - Justification: 
```

### 5.3 Resource Utilization
```yaml
Resource Usage:
  GPU Memory:
    - Peak Usage: # GB
    - Average Usage: # GB
    
  CPU Utilization:
    - Peak: # %
    - Average: # %
    
  Disk I/O:
    - Read Throughput: # MB/s
    - Write Throughput: # MB/s
```

### 5.4 Benchmark Results
<!-- Include relevant benchmark results -->
```python
# Benchmark configuration and results
"""
Benchmark Setup:
- Hardware: # e.g., NVIDIA A100 40GB
- Batch Size: 
- Sequence Length: 
- Number of Runs: 

Results:
- Throughput: X samples/second
- Latency P50: X ms
- Latency P95: X ms
- Latency P99: X ms
"""
```

## 6. COMPATIBILITY AND DEPENDENCIES

### 6.1 Backward Compatibility
<!-- Assess backward compatibility impact -->
- [ ] **Fully Backward Compatible**: No breaking changes
- [ ] **Backward Compatible with Deprecation**: Old API deprecated but functional
- [ ] **Breaking Changes**: Requires migration
  - Migration guide provided: [ ] Yes [ ] No
  - Affected components: 
  - Migration complexity: Low/Medium/High

### 6.2 Python Version Compatibility
<!-- Tested Python versions -->
- [ ] Python 3.8 (Minimum supported)
- [ ] Python 3.9
- [ ] Python 3.10 (Recommended)
- [ ] Python 3.11
- [ ] Python 3.12

### 6.3 Dependency Changes
<!-- List all dependency modifications -->
```yaml
Added Dependencies:
  - package_name>=X.Y.Z  # Purpose/Justification

Updated Dependencies:
  - package_name: X.Y.Z -> A.B.C  # Reason for update

Removed Dependencies:
  - package_name  # Reason for removal
```

### 6.4 Hardware Requirements
<!-- Specify any new hardware requirements -->
```yaml
Minimum Requirements:
  - GPU Memory: # GB
  - System RAM: # GB
  - Disk Space: # GB
  
Recommended Requirements:
  - GPU Memory: # GB
  - System RAM: # GB
  - Disk Space: # GB
```

## 7. DOCUMENTATION

### 7.1 Documentation Updates
<!-- List all documentation changes -->
- [ ] **API Documentation**: Docstrings added/updated
- [ ] **User Guide**: Usage instructions updated
- [ ] **README**: Project documentation updated
- [ ] **Architecture Docs**: System design documentation updated
- [ ] **Configuration Guide**: Configuration documentation updated
- [ ] **Tutorials**: Example notebooks/scripts updated
- [ ] **CHANGELOG**: Version history updated
- [ ] **Migration Guide**: Breaking change documentation

### 7.2 Code Documentation Quality
<!-- Confirm documentation standards -->
- [ ] All public functions have comprehensive docstrings
- [ ] Complex algorithms include explanatory comments
- [ ] Type hints provided for all function signatures
- [ ] Examples included in docstrings where appropriate
- [ ] Mathematical formulations documented where applicable
- [ ] References to papers/resources included

### 7.3 Documentation Examples
```python
def example_function(input_tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """
    Brief description of the function.
    
    Detailed explanation of the algorithm, including mathematical formulation
    if applicable. References to academic papers.
    
    Args:
        input_tensor: Shape [batch_size, sequence_length, hidden_dim]
            Input tensor description
        config: Configuration dictionary containing:
            - key1: Description
            - key2: Description
    
    Returns:
        torch.Tensor: Shape [batch_size, num_classes]
            Output tensor description
    
    Raises:
        ValueError: If input dimensions are incorrect
        RuntimeError: If CUDA out of memory
    
    Examples:
        >>> tensor = torch.randn(32, 128, 768)
        >>> config = {'key1': value1, 'key2': value2}
        >>> output = example_function(tensor, config)
        >>> assert output.shape == (32, 4)
    
    References:
        [1] Author et al. (2024). "Paper Title". Conference.
    """
    # Implementation
    pass
```

## 8. CODE QUALITY ASSURANCE

### 8.1 Code Style Compliance
<!-- Verify code style standards -->
- [ ] **PEP 8 Compliance**: Python style guide followed
- [ ] **Black Formatting**: Code formatted with Black
- [ ] **isort**: Imports sorted correctly
- [ ] **Type Hints**: Static typing used appropriately
- [ ] **Linting**: No linting errors (flake8, pylint)
- [ ] **Security**: No security vulnerabilities (bandit)

### 8.2 Code Quality Metrics
```yaml
Static Analysis Results:
  Pylint Score: /10
  Cyclomatic Complexity: 
  Maintainability Index: 
  Technical Debt Ratio: %
  Code Duplication: %
```

### 8.3 Design Principles
<!-- Confirm adherence to design principles -->
- [ ] **SOLID Principles**: Applied where appropriate
- [ ] **DRY Principle**: No unnecessary code duplication
- [ ] **KISS Principle**: Solution is as simple as possible
- [ ] **YAGNI Principle**: No unnecessary features added
- [ ] **Separation of Concerns**: Clear module boundaries

### 8.4 Error Handling
<!-- Describe error handling approach -->
```python
# Error handling strategy
"""
1. Input Validation:
   - Parameters validated at entry points
   - Clear error messages provided
   
2. Exception Handling:
   - Specific exceptions caught and handled
   - Graceful degradation where applicable
   
3. Logging:
   - Appropriate log levels used
   - Sufficient context in log messages
"""
```

## 9. SECURITY AND PRIVACY

### 9.1 Security Checklist
- [ ] No hardcoded credentials or API keys
- [ ] No sensitive data in logs or error messages
- [ ] Input validation implemented for user inputs
- [ ] SQL injection prevention (if applicable)
- [ ] Path traversal prevention (if applicable)
- [ ] Dependency vulnerabilities checked

### 9.2 Privacy Considerations
- [ ] No PII (Personally Identifiable Information) logged
- [ ] Data anonymization implemented where needed
- [ ] GDPR compliance maintained (if applicable)

## 10. RESEARCH CONTRIBUTION (if applicable)

### 10.1 Academic Context
```yaml
Research Contribution:
  Paper Reference: # Authors (Year). "Title". Venue.
  Algorithm Name: 
  Novel Contributions:
    - Contribution 1
    - Contribution 2
  
  Theoretical Justification:
    - Mathematical proof provided: Yes/No
    - Complexity analysis provided: Yes/No
```

### 10.2 Experimental Validation
```yaml
Experiments:
  Datasets Tested:
    - AG News: Results
    - Additional Dataset: Results
  
  Baselines Compared:
    - Baseline 1: Improvement %
    - Baseline 2: Improvement %
  
  Statistical Significance:
    - Test Used: # e.g., paired t-test
    - p-value: 
    - Effect Size: 
```

### 10.3 Reproducibility
```yaml
Reproducibility Information:
  Random Seeds: # List all seeds used
  Hardware Specifications:
    - GPU: 
    - CUDA Version: 
    - cuDNN Version: 
  
  Hyperparameters:
    - Learning Rate: 
    - Batch Size: 
    - Epochs: 
    - Other: 
```

## 11. DEPLOYMENT CONSIDERATIONS

### 11.1 Deployment Readiness
- [ ] Production configuration files updated
- [ ] Environment variables documented
- [ ] Docker images updated and tested
- [ ] Kubernetes manifests updated (if applicable)
- [ ] Database migrations prepared (if applicable)

### 11.2 Rollback Plan
```yaml
Rollback Strategy:
  - Rollback Complexity: Low/Medium/High
  - Rollback Steps:
    1. Step 1
    2. Step 2
  - Data Migration Reversible: Yes/No
  - Estimated Rollback Time: 
```

## 12. REVIEW CHECKLIST

### 12.1 Author Checklist
<!-- Complete before requesting review -->
- [ ] Code compiles without warnings
- [ ] All tests pass locally
- [ ] Documentation is complete and accurate
- [ ] Code follows project style guidelines
- [ ] No debugging code or print statements left
- [ ] Sensitive information removed
- [ ] Performance impact assessed
- [ ] Security implications considered
- [ ] Backward compatibility verified
- [ ] CHANGELOG updated

### 12.2 Reviewer Checklist
<!-- For reviewers to consider -->
- [ ] Code logic and correctness
- [ ] Algorithm efficiency
- [ ] Test coverage adequacy
- [ ] Documentation completeness
- [ ] Code style consistency
- [ ] Security best practices
- [ ] Performance implications
- [ ] Architectural alignment
- [ ] Research validity (if applicable)

## 13. ADDITIONAL INFORMATION

### 13.1 Screenshots/Visualizations
<!-- Include relevant visualizations -->
<!-- Attach images demonstrating UI changes, model outputs, or performance graphs -->

### 13.2 Future Work
```yaml
Follow-up Tasks:
  - Task 1: Description
  - Task 2: Description
  
Known Limitations:
  - Limitation 1: Description
  - Limitation 2: Description
  
Planned Improvements:
  - Improvement 1: Timeline
  - Improvement 2: Timeline
```

### 13.3 Acknowledgments
<!-- Credit contributors, reviewers, or resources -->
```
Acknowledgments:
- 
- 
```

### 13.4 Notes for Reviewers
<!-- Any specific areas where you'd like feedback -->
```
Review Focus Areas:
- 
- 
```

---

<!--
================================================================================
REVIEW PROCESS NOTES
================================================================================
Thank you for your contribution to the AG News Classification system.

Review Process:
1. Automated CI/CD checks will run upon submission
2. Code review by project maintainers
3. Feedback and revision cycle
4. Approval and merge upon satisfactory review

Expected Review Timeline:
- Initial review: Within 48-72 hours
- Subsequent reviews: Within 24-48 hours

For urgent reviews, please contact @VoHaiDung with justification.

Merge Criteria:
- All CI/CD checks pass
- At least one maintainer approval
- No unresolved review comments
- Documentation complete
- Tests adequate
================================================================================
-->
