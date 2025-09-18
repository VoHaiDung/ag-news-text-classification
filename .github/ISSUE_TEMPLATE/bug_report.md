---
name: Bug Report
about: Create a report to help improve the AG News Classification system
title: '[BUG] '
labels: 'bug, needs-triage'
assignees: ''

---

<!--
================================================================================
BUG REPORT TEMPLATE - AG NEWS CLASSIFICATION SYSTEM
================================================================================
This template follows software engineering best practices and academic standards
for bug reporting in machine learning research projects.

References:
- IEEE Standard for Software Test Documentation (IEEE 829-2008)
- ACM Guidelines for Reporting Computational Research
- GitHub Issue Reporting Best Practices

Please provide comprehensive information to facilitate efficient debugging and
resolution. Incomplete reports may delay the investigation process.
================================================================================
-->

## 1. BUG SUMMARY

### 1.1 Brief Description
<!-- Provide a concise, one-line summary of the bug -->

### 1.2 Affected Component
<!-- Specify which component is affected -->
- [ ] Data Loading/Processing (`src/data/`)
- [ ] Model Architecture (`src/models/`)
- [ ] Training Pipeline (`src/training/`)
- [ ] Evaluation System (`src/evaluation/`)
- [ ] Inference Pipeline (`src/inference/`)
- [ ] Configuration System (`configs/`)
- [ ] API/Services (`src/api/`, `src/services/`)
- [ ] Deployment (`deployment/`)
- [ ] Documentation (`docs/`)
- [ ] Other: <!-- Please specify -->

### 1.3 Bug Classification
<!-- Classify the type of bug -->
- [ ] Functional Error (incorrect output/behavior)
- [ ] Performance Degradation
- [ ] Memory Leak/Resource Issue
- [ ] Compatibility Issue
- [ ] Configuration Error
- [ ] Documentation Inconsistency
- [ ] Build/Installation Failure
- [ ] Other: <!-- Please specify -->

## 2. REPRODUCTION INFORMATION

### 2.1 Current Behavior
<!-- Describe the actual behavior observed when the bug occurs -->
```
Actual behavior description:
```

### 2.2 Expected Behavior
<!-- Describe what should happen instead -->
```
Expected behavior description:
```

### 2.3 Steps to Reproduce
<!-- Provide detailed, numbered steps to reproduce the issue -->
1. Environment preparation:
   ```bash
   # Commands to set up environment
   ```

2. Data preparation:
   ```bash
   # Commands or steps to prepare data
   ```

3. Code execution:
   ```python
   # Minimal code to reproduce the issue
   ```

4. Observed error/issue:
   ```bash
   # Description of what happens
   ```

### 2.4 Minimal Reproducible Example
<!-- Provide the smallest possible code example that demonstrates the bug -->
```python
"""
Minimal reproducible example for bug reproduction.
Remove any unnecessary code that is not related to the bug.
"""
import sys
import os

# Add project root to path
sys.path.append('/path/to/ag-news-text-classification')

# Import necessary modules
from src.data.datasets.ag_news import AGNewsDataset
# Add other necessary imports

# Code that triggers the bug
def reproduce_bug():
    """Function that reproduces the reported bug."""
    # Your code here
    pass

if __name__ == "__main__":
    reproduce_bug()
```

## 3. ENVIRONMENT SPECIFICATIONS

### 3.1 System Information
```yaml
# System configuration details
Operating System: # e.g., Ubuntu 22.04 LTS, Windows 11, macOS Ventura 13.0
OS Version: 
Architecture: # e.g., x86_64, arm64
CPU Model: 
RAM: # e.g., 32GB
GPU Model: # e.g., NVIDIA RTX 3090 24GB
CUDA Version: # if applicable
cuDNN Version: # if applicable
```

### 3.2 Python Environment
```yaml
Python Version: # e.g., 3.10.12
Virtual Environment: # e.g., venv, conda, poetry
Package Manager: # e.g., pip, conda
```

### 3.3 Dependency Versions
<!-- Execute: pip freeze | grep -E "torch|transformers|numpy|pandas|scikit-learn|pyyaml|datasets" -->
```
torch==
transformers==
numpy==
pandas==
scikit-learn==
pyyaml==
datasets==
accelerate==
tokenizers==
```

### 3.4 Installation Method
<!-- How was the project installed? -->
- [ ] Standard installation: `pip install -r requirements.txt`
- [ ] Development installation: `pip install -e .`
- [ ] Docker container: `docker pull ghcr.io/vohaidung/ag-news-classification`
- [ ] Conda environment: `conda env create -f environment.yml`
- [ ] Manual installation
- [ ] Other: <!-- Please specify -->

## 4. ERROR INFORMATION

### 4.1 Error Messages
<!-- Copy the complete error message -->
```
[Paste complete error message here, including any warnings]
```

### 4.2 Stack Trace
<!-- Provide the full stack trace if available -->
```python
Traceback (most recent call last):
  File "...", line X, in <module>
    ...
[Paste complete stack trace here]
```

### 4.3 Log Files
<!-- Attach or provide links to relevant log files -->
- Training logs: <!-- Path or attachment -->
- System logs: <!-- Path or attachment -->
- Debug logs: <!-- Path or attachment -->

### 4.4 Error Timing
- When did the error first occur? <!-- Date/time if known -->
- Is the error consistent or intermittent?
  - [ ] Consistent (happens every time)
  - [ ] Intermittent (happens occasionally)
  - [ ] Frequency: <!-- e.g., 3 out of 5 runs -->

## 5. DATA AND MODEL CONTEXT

### 5.1 Dataset Information
```yaml
Dataset Name: # e.g., AG News, Custom Dataset
Dataset Version: 
Dataset Size: # Number of samples
Data Format: # e.g., CSV, JSON, Parquet
Preprocessing Applied: # List preprocessing steps
Data Split: # e.g., train/val/test ratio
```

### 5.2 Model Configuration
```yaml
Model Architecture: # e.g., DeBERTa-v3-large, RoBERTa-large
Model Checkpoint: # e.g., microsoft/deberta-v3-large
Configuration File: # e.g., configs/models/single/deberta_v3_xlarge.yaml
Custom Modifications: # Any modifications to the standard model
```

### 5.3 Training Configuration (if applicable)
```yaml
Batch Size: 
Learning Rate: 
Optimizer: 
Scheduler: 
Training Steps/Epochs: 
Mixed Precision: # Yes/No
Distributed Training: # Yes/No
```

## 6. DIAGNOSTIC INFORMATION

### 6.1 Attempted Solutions
<!-- List all solutions you have tried -->
- [ ] Verified all dependencies are correctly installed
- [ ] Checked documentation and existing issues
- [ ] Tested with different Python versions
- [ ] Tested on different hardware (CPU vs GPU)
- [ ] Cleared cache and temporary files
- [ ] Reinstalled the project from scratch
- [ ] Tested with different data samples
- [ ] Reduced batch size/model size
- [ ] Other attempts: <!-- Please list -->

### 6.2 Workarounds
<!-- Describe any temporary workarounds you've found -->
```
Workaround description (if any):
```

### 6.3 Related Issues
<!-- Link to related issues or discussions -->
- Related to issue #
- Similar to issue #
- Discussed in #

## 7. IMPACT ASSESSMENT

### 7.1 Severity Level
<!-- Assess the severity of this bug -->
- [ ] **Critical**: System unusable, data loss, or security issue
- [ ] **High**: Major functionality broken, no workaround available
- [ ] **Medium**: Important functionality impaired, workaround available
- [ ] **Low**: Minor issue, cosmetic problem, or enhancement

### 7.2 Affected Users
<!-- Who is impacted by this bug? -->
- [ ] All users
- [ ] Specific configuration users
- [ ] Development environment only
- [ ] Production environment only
- [ ] Specific hardware/OS users

### 7.3 Business/Research Impact
<!-- Describe the impact on your work or research -->
```
Impact description:
```

## 8. ADDITIONAL CONTEXT

### 8.1 Screenshots/Visualizations
<!-- If applicable, add screenshots, plots, or diagrams -->
<!-- Drag and drop images here or provide links -->

### 8.2 Performance Metrics (if applicable)
<!-- For performance-related bugs -->
| Metric | Expected | Actual | Difference |
|--------|----------|--------|------------|
| Training Speed (samples/sec) | | | |
| Inference Latency (ms) | | | |
| Memory Usage (GB) | | | |
| Model Accuracy (%) | | | |

### 8.3 Relevant Code Sections
<!-- Point to specific code sections if you've identified the problem area -->
```python
# File: src/path/to/file.py
# Lines: XXX-YYY
# Suspected problematic code
```

### 8.4 References
<!-- Academic papers, documentation, or other references relevant to this bug -->
1. 
2. 

## 9. PROPOSED SOLUTION (OPTIONAL)

### 9.1 Root Cause Analysis
<!-- If you have insights into the root cause -->
```
Potential root cause:
```

### 9.2 Suggested Fix
<!-- If you have suggestions for fixing the bug -->
```python
# Proposed code changes or approach
```

### 9.3 Implementation Considerations
<!-- Any considerations for the fix implementation -->
- Performance implications:
- Backward compatibility:
- Testing requirements:

## 10. CHECKLIST

### 10.1 Reporter Checklist
<!-- Please check all completed items before submission -->
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a clear and descriptive title
- [ ] I have included all required environment information
- [ ] I have provided a minimal reproducible example
- [ ] I have included complete error messages and stack traces
- [ ] I have tested with the latest version of the main branch
- [ ] I have read the contributing guidelines
- [ ] I have removed any sensitive information from logs/code

### 10.2 Contribution Willingness
- [ ] I am willing to submit a Pull Request to fix this issue
- [ ] I am available to provide additional information if needed
- [ ] I can help test the fix once implemented

---

<!--
================================================================================
SUBMISSION NOTES
================================================================================
Thank you for taking the time to report this bug. Your detailed report helps
maintain the quality and reliability of the AG News Classification system.

After submission:
1. The issue will be triaged by maintainers
2. Additional information may be requested
3. The issue will be prioritized based on severity and impact
4. Updates will be provided as investigation progresses

For urgent issues, please mention @VoHaiDung in the comments.
================================================================================
-->
