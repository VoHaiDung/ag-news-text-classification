# AG News Text Classification - Lock Files

## Project Information

- **Project**: AG News Text Classification (ag-news-text-classification)
- **Author**: Võ Hải Dũng
- **Email**: vohaidung.work@gmail.com
- **License**: MIT

## Overview

This directory contains locked dependency files with exact package versions for reproducible installations across different development, testing, and production environments.

## Purpose and Benefits

Lock files provide critical guarantees for software development:

### Reproducibility
- Ensures identical package versions across all team members
- Guarantees consistent behavior in different environments
- Enables exact replication of research results
- Facilitates debugging by eliminating version-related variability

### Stability
- Prevents unexpected breaking changes from automatic updates
- Protects against incompatible dependency upgrades
- Reduces "works on my machine" problems
- Provides predictable deployment outcomes

### Security
- Creates auditable record of exact package versions
- Enables precise vulnerability tracking
- Facilitates targeted security patching
- Supports compliance and regulatory requirements

### CI/CD Integration
- Ensures consistent builds in automated pipelines
- Reduces flaky tests due to dependency changes
- Enables reliable rollbacks to known-good states
- Supports reproducible artifacts

### Production Deployment
- Guarantees development-production parity
- Minimizes deployment risks
- Enables gradual rollouts with confidence
- Supports disaster recovery procedures

## Available Lock Files

### Core Lock Files

| Lock File | Packages | Size | Installation Time | Primary Use Case |
|-----------|----------|------|-------------------|------------------|
| `base.lock` | ~150 | ~2GB | 5-10 min | Core dependencies, minimal setup |
| `ml.lock` | ~290 | ~8GB | 15-20 min | Machine learning training |
| `llm.lock` | ~320 | ~10GB | 20-30 min | Large language model fine-tuning |
| `all.lock` | ~415 | ~15GB | 30-60 min | Complete development environment |

### Lock File Hierarchy and Dependencies

```
base.lock
├── Core PyTorch (CPU/GPU)
├── Transformers ecosystem
├── Scientific computing (NumPy, SciPy, Pandas)
├── Configuration management
└── Essential utilities

ml.lock (extends base.lock)
├── All base.lock packages
├── Parameter-efficient fine-tuning (LoRA, QLoRA)
├── Classical ML (XGBoost, LightGBM, CatBoost)
├── Ensemble methods
├── Hyperparameter optimization
├── Experiment tracking
├── Data augmentation
└── Model compression

llm.lock (extends ml.lock)
├── All ml.lock packages
├── LLM training frameworks (TRL)
├── Efficient attention (Flash Attention, xFormers)
├── LLM inference (vLLM)
├── Prompt engineering (LangChain, LlamaIndex)
├── Vector databases (ChromaDB, FAISS)
├── LLM APIs and embeddings
└── Document processing

all.lock (extends llm.lock)
├── All llm.lock packages
├── Web UI frameworks (Streamlit, Gradio, Dash)
├── Advanced visualization
├── Development tools (testing, linting, debugging)
├── Documentation generation (Sphinx, MkDocs)
├── Data scraping and collection
├── Database support
├── API frameworks (FastAPI)
├── Monitoring and observability
├── Security scanning tools
└── Robustness testing
```

## Usage Scenarios

### When to Use Lock Files

#### Production Deployment
Always use lock files in production to ensure stability and reproducibility.

```bash
pip install -r requirements/lock/all.lock
python scripts/setup/verify_installation.py
```

#### CI/CD Pipelines
Use lock files for consistent automated builds and tests.

```yaml
# GitHub Actions example
- name: Install dependencies
  run: pip install -r requirements/lock/ml.lock
```

#### Research Reproducibility
Use lock files to ensure exact replication of experimental results.

```bash
# Reproduce experiment from paper
pip install -r requirements/lock/llm.lock
python experiments/sota_experiments/phase5_ultimate_sota.py
```

#### Bug Investigation
Use lock files to recreate the exact environment where a bug occurred.

```bash
# Reproduce bug environment
git checkout bug-report-commit
pip install -r requirements/lock/all.lock
pytest tests/test_specific_bug.py
```

#### Offline Installation
Pre-download lock file dependencies for air-gapped systems.

```bash
# On online machine
pip download -r requirements/lock/ml.lock -d packages/

# Transfer packages/ to offline machine
pip install --no-index --find-links packages/ -r requirements/lock/ml.lock
```

### When to Use Regular Requirements

#### Local Development
Use regular requirements for flexibility during active development.

```bash
pip install -r requirements/ml.txt
```

#### Experimenting with New Packages
Test new package versions without affecting lock files.

```bash
pip install -r requirements/ml.txt
pip install new-experimental-package
```

#### Security Updates
Quickly adopt latest security patches.

```bash
pip install -r requirements/ml.txt --upgrade
```

#### Dependency Resolution
Let pip resolve latest compatible versions.

```bash
pip install -r requirements/all_local.txt
```

## Installation Instructions

### Standard Installation from Lock File

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install from lock file
pip install -r requirements/lock/all.lock
```

### Platform-Specific Installation

#### Linux with CUDA 11.8 (Recommended)

```bash
# Install PyTorch with CUDA 11.8 first
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Install rest from lock file
pip install -r requirements/lock/ml.lock
```

#### CUDA 12.1

```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Install from source requirements (not lock file)
pip install -r requirements/ml.txt
```

#### CPU-Only

```bash
# Install CPU version of PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cpu

# Install rest
pip install -r requirements/lock/ml.lock
```

#### Apple Silicon (M1/M2/M3)

```bash
# Default installation uses MPS backend
pip install -r requirements/lock/ml.lock
```

#### Windows

```bash
# Some Linux-only packages will be skipped
pip install -r requirements/ml.txt  # Use source requirements
```

### Verification After Installation

```bash
# Verify all packages installed correctly
python scripts/setup/verify_installation.py

# Check for dependency conflicts
pip check

# Verify critical imports
python -c "import torch; import transformers; import peft; print('OK')"

# Check CUDA availability (if GPU)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run health checks
python src/core/health/health_checker.py --comprehensive
```

## Generating Lock Files

### Manual Generation Process

```bash
# Create temporary clean environment
python -m venv /tmp/lock_env
source /tmp/lock_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install from source requirements
pip install -r requirements/base.txt

# Generate lock file
pip freeze > requirements/lock/base.lock

# Add header documentation
# (Edit file to add project information and documentation)

# Cleanup
deactivate
rm -rf /tmp/lock_env
```

### Automated Generation Script

```bash
# Use provided automation script
bash scripts/ci/update_lock_files.sh

# Or for specific lock file
bash scripts/ci/update_lock_files.sh base
bash scripts/ci/update_lock_files.sh ml
bash scripts/ci/update_lock_files.sh llm
bash scripts/ci/update_lock_files.sh all
```

The automated script performs:
1. Creates isolated virtual environments
2. Installs from source requirements
3. Freezes exact versions
4. Adds documentation headers
5. Runs security audits
6. Validates installations
7. Generates compatibility reports
8. Cleans up temporary files

### Lock File Generation Best Practices

- Generate on clean virtual environment
- Use consistent Python version (3.10.12 recommended)
- Document generation date and platform
- Include comprehensive header comments
- Run security audit after generation
- Test installation before committing
- Update CHANGELOG.md with changes

## Lock File Format and Structure

### Header Section

Every lock file begins with a comprehensive header:

```text
# ============================================================================
# Locked [Type] Requirements for AG News Text Classification
# ============================================================================
# Project: AG News Text Classification (ag-news-text-classification)
# Description: [Purpose of this lock file]
# Author: Võ Hải Dũng
# Email: vohaidung.work@gmail.com
# License: MIT
# Generated: YYYY-MM-DD
# Python: 3.10.12
# Platform: linux-x86_64
# CUDA: 11.8 (if applicable)
# ============================================================================
```

### Package Sections

Packages are organized by category with descriptive comments:

```text
# ----------------------------------------------------------------------------
# Category Name
# ----------------------------------------------------------------------------
package-name==exact.version
another-package==exact.version+build_tag
```

### Platform-Specific Builds

PyTorch packages include build tags:

```text
torch==2.1.2+cu118        # CUDA 11.8 build
torchvision==0.16.2+cu118 # CUDA 11.8 build
torchaudio==2.1.2+cu118   # CUDA 11.8 build
```

### Transitive Dependencies

Auto-installed dependencies are documented:

```text
# ----------------------------------------------------------------------------
# Transitive Dependencies (Auto-Installed)
# ----------------------------------------------------------------------------
# Note: Automatically installed as dependencies of packages above
dependency-of-main-package==version
another-dependency==version
```

### Documentation Section

Each lock file includes comprehensive installation and usage documentation at the end.

## Platform Compatibility

### Tested Platforms

| Platform | Python | CUDA | Status | Notes |
|----------|--------|------|--------|-------|
| Ubuntu 20.04 LTS | 3.8, 3.9, 3.10 | 11.8 | Fully Tested | Recommended |
| Ubuntu 22.04 LTS | 3.10, 3.11 | 11.8 | Fully Tested | Recommended |
| Debian 11 | 3.9, 3.10 | 11.8 | Tested | Compatible |
| Windows 10/11 | 3.10 | 11.8 | Partially Tested | Some packages unavailable |
| macOS 12 Monterey | 3.10 | N/A | CPU Only | Intel |
| macOS 13 Ventura | 3.10 | N/A | CPU+MPS | Apple Silicon |
| Google Colab | 3.10 | 11.8 | Fully Tested | Free tier compatible |
| Kaggle Kernels | 3.10 | 11.8 | Fully Tested | TPU support |

### Known Platform-Specific Issues

#### Windows
- **DeepSpeed**: Not supported (Linux only)
- **Horovod**: Requires MPI, may fail during installation
- **Flash Attention**: Not available
- **Recommendation**: Use WSL2 for full compatibility

#### macOS
- **CUDA**: Not available (use CPU or MPS backend)
- **Flash Attention**: Not supported
- **vLLM**: Not supported
- **Recommendation**: Use for CPU inference or development only

#### ARM64 Architecture
- **Binary packages**: May have different versions
- **Compilation**: Some packages require building from source
- **Flash Attention**: Limited support

#### Python 3.12
- **Package availability**: Some packages not yet compatible
- **Recommendation**: Use Python 3.10 or 3.11

### Compatibility Matrix

For detailed compatibility information, see:
- `configs/compatibility_matrix.yaml`
- Platform-specific guides in `docs/platform_guides/`

## Security and Auditing

### Security Audit Schedule

| Audit Type | Frequency | Tools Used |
|------------|-----------|------------|
| Automated Scan | Weekly | GitHub Dependabot |
| Manual Review | Monthly | pip-audit, safety, snyk |
| Comprehensive Audit | Quarterly | Full security assessment |
| Critical Updates | As needed | Immediate patching |

### Running Security Audits

```bash
# Install audit tools
pip install pip-audit safety

# Run pip-audit
pip-audit -r requirements/lock/all.lock --desc

# Run safety check
safety check -r requirements/lock/all.lock --json

# Generate audit report
pip-audit -r requirements/lock/all.lock --format json > audit_report.json
safety check -r requirements/lock/all.lock --json > safety_report.json

# Online scanning with snyk
snyk test --file=requirements/lock/all.lock
```

### Security Best Practices

1. **Regular Updates**: Update lock files monthly for security patches
2. **Vulnerability Monitoring**: Enable GitHub Dependabot alerts
3. **Audit Before Deployment**: Always audit before production deployment
4. **Document Vulnerabilities**: Track known issues in security log
5. **Emergency Patching**: Have process for critical vulnerability response
6. **Secrets Management**: Never commit secrets to lock files
7. **SBOM Generation**: Generate Software Bill of Materials for compliance

### Last Security Audit

- **Date**: 2025-09-19
- **Critical Vulnerabilities**: 0
- **High Vulnerabilities**: 0
- **Medium Vulnerabilities**: 0
- **Low Vulnerabilities**: 0
- **Tools**: pip-audit 2.6.1, safety 3.0.1, snyk, GitHub Dependabot

## Maintenance and Updates

### Regular Maintenance Schedule

| Activity | Frequency | Responsible |
|----------|-----------|-------------|
| Security monitoring | Weekly | Automated + Review |
| Minor updates | Monthly | Maintainer |
| Major updates | Quarterly | Team review |
| Lock file regeneration | After req changes | Developer |
| Documentation updates | As needed | Contributor |

### Update Process Workflow

#### 1. Preparation Phase

```bash
# Create backup
cp requirements/lock/*.lock requirements/lock/backup/

# Create feature branch
git checkout -b update-dependencies-2025-09

# Review release notes
# Check security advisories
# Plan breaking changes
```

#### 2. Update Source Requirements

```bash
# Update version constraints in requirements/*.txt
# Example: transformers>=4.36.0,<4.41.0 -> transformers>=4.38.0,<4.43.0
```

#### 3. Generate New Lock Files

```bash
# Run automated update script
bash scripts/ci/update_lock_files.sh

# Or manual generation
python -m venv temp_env
source temp_env/bin/activate
pip install -r requirements/all_local.txt
pip freeze > requirements/lock/all.lock
deactivate
rm -rf temp_env
```

#### 4. Testing Phase

```bash
# Install new lock file
pip install -r requirements/lock/all.lock

# Run comprehensive tests
pytest tests/ -v --cov

# Run benchmarks
python experiments/benchmarks/accuracy_benchmark.py
python experiments/benchmarks/speed_benchmark.py

# Run smoke tests
pytest tests/smoke/ -v
```

#### 5. Security Validation

```bash
# Run security audit
pip-audit -r requirements/lock/all.lock
safety check -r requirements/lock/all.lock
```

#### 6. Documentation

```markdown
# Update CHANGELOG.md

## [1.1.0] - 2025-10-19

### Changed
- Updated all dependencies to latest compatible versions
- PyTorch 2.1.2 -> 2.2.0
- Transformers 4.37.2 -> 4.38.0
- Lock files regenerated with updated versions

### Security
- Fixed CVE-2024-XXXXX in package-name
- Updated cryptography to patch vulnerability

### Breaking Changes
- None

### Migration Notes
- No manual migration required
- Regenerate virtual environment recommended
```

#### 7. Commit and Release

```bash
# Commit changes
git add requirements/lock/*.lock CHANGELOG.md
git commit -m "chore: update dependencies to 2025-09"

# Create pull request
git push origin update-dependencies-2025-09
# Review, approve, merge

# Tag release
git tag v1.2.0
git push origin v1.2.0

# Generate release notes on GitHub
```

### Handling Breaking Changes

When updates introduce breaking changes:

#### 1. Version Increment

Follow semantic versioning:
- **Patch** (1.0.x): Bug fixes, security patches
- **Minor** (1.x.0): New features, backward compatible
- **Major** (x.0.0): Breaking changes

#### 2. Migration Guide

Create migration guide in `docs/migrations/vX.X.X.md`:

```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes

### Transformers 5.0 API Changes

**Impact**: High

**Change**: `AutoModel.from_pretrained()` now requires explicit trust

**Before** (v1.x):
```python
model = AutoModel.from_pretrained("model_name")
```

**After** (v2.0):
```python
model = AutoModel.from_pretrained(
    "model_name",
    trust_remote_code=True  # Required in v5.0
)
```

**Action Required**: Update all model loading code
```

#### 3. Deprecation Warnings

Provide warnings in previous version:

```python
import warnings
warnings.warn(
    "AutoModel.from_pretrained() will require trust_remote_code=True in v2.0",
    DeprecationWarning,
    stacklevel=2
)
```

#### 4. Compatibility Layer

Provide backward compatibility when possible:

```python
def load_model(model_name, trust_remote_code=None):
    if trust_remote_code is None:
        # Auto-detect based on version
        import transformers
        if transformers.__version__ >= "5.0.0":
            trust_remote_code = True
    return AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
```

## Troubleshooting

### Common Installation Issues

#### Issue: Package Installation Fails

```text
ERROR: Could not find a version that satisfies the requirement package==version
```

**Diagnosis**:
- Package not available for your platform
- Python version incompatibility
- Corrupted pip cache

**Solutions**:
```bash
# Check Python version
python --version  # Should be 3.8-3.11

# Clear pip cache
pip cache purge

# Try installing from source requirements
pip install -r requirements/ml.txt

# Check platform compatibility
python -c "import platform; print(platform.system(), platform.machine())"
```

#### Issue: CUDA Version Mismatch

```text
RuntimeError: CUDA version mismatch: runtime version XX.X, driver version YY.Y
```

**Diagnosis**:
- PyTorch compiled for different CUDA version
- CUDA driver/runtime mismatch

**Solutions**:
```bash
# Check CUDA versions
nvcc --version          # CUDA Toolkit version
nvidia-smi              # CUDA Driver version

# Install matching PyTorch
# For CUDA 11.8:
pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: Out of Disk Space

```text
ERROR: No space left on device
```

**Solutions**:
```bash
# Check disk space
df -h

# Clear pip cache
pip cache purge

# Clear conda cache (if applicable)
conda clean --all -y

# Clear Docker images (if applicable)
docker system prune -a

# Clear old virtual environments
rm -rf old_venv/
```

#### Issue: Compilation Errors

```text
ERROR: Failed building wheel for package-name
```

**Solutions**:
```bash
# Install build tools

# Ubuntu/Debian:
sudo apt update
sudo apt install build-essential python3-dev

# macOS:
xcode-select --install

# Windows:
# Install Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/
```

#### Issue: SSL Certificate Errors

```text
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions**:
```bash
# Update certificates
pip install --upgrade certifi

# Use trusted hosts (temporary, not for production)
pip install -r requirements/lock/ml.lock \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

#### Issue: Memory Error During Installation

```text
MemoryError: Unable to allocate array
```

**Solutions**:
```bash
# Close other applications
# Increase swap space (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install in smaller chunks
pip install -r requirements/lock/base.lock
pip install -r requirements/lock/ml.lock  # Incrementally
```

### Verification Issues

#### Issue: Import Errors After Installation

```python
ImportError: cannot import name 'X' from 'package'
```

**Solutions**:
```bash
# Verify installation
pip list | grep package-name

# Check for multiple installations
pip show package-name

# Reinstall in clean environment
python -m venv new_venv
source new_venv/bin/activate
pip install -r requirements/lock/ml.lock
```

#### Issue: Version Conflicts

```text
ERROR: package-a requires package-b>=2.0, but package-b 1.5 is installed
```

**Solutions**:
```bash
# Use clean virtual environment
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements/lock/all.lock

# Check for conflicts
pip check
```

### Platform-Specific Issues

#### Windows: Package Not Available

Some packages are Linux-only. Use source requirements instead:

```bash
# Skip platform-specific packages
pip install -r requirements/ml.txt
```

#### macOS: Flash Attention Not Available

Flash Attention is Linux-only. xFormers will be used as fallback automatically.

#### Colab/Kaggle: GPU Out of Memory

Use platform-optimized configs:

```bash
pip install -r requirements/colab.txt  # Or kaggle.txt
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache pip packages
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements/lock/ml.lock') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements/lock/ml.lock
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov
      
      - name: Security audit
        run: |
          pip install pip-audit safety
          pip-audit -r requirements/lock/ml.lock
          safety check -r requirements/lock/ml.lock
```

### GitLab CI

```yaml
test:
  image: python:3.10
  
  cache:
    paths:
      - .cache/pip
  
  before_script:
    - pip install -r requirements/lock/ml.lock
  
  script:
    - pytest tests/ -v --cov
    - pip-audit -r requirements/lock/ml.lock
  
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy lock file
COPY requirements/lock/ml.lock .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r ml.lock

# Copy application
COPY . .

# Run
CMD ["python", "train.py"]
```

## FAQ

### Q: Why use lock files instead of version ranges?

**A**: Lock files provide exact reproducibility:
- Same versions across all environments
- Consistent CI/CD builds
- Precise security audit trail
- Predictable production deployments
- Easier debugging (eliminate version variables)

### Q: How often should I update lock files?

**A**: 
- **Security patches**: Immediately for critical vulnerabilities
- **Regular updates**: Monthly for new features and fixes
- **Major versions**: After thorough testing, typically quarterly
- **After dependency changes**: Whenever requirements/*.txt changes

### Q: Can I use lock files across different platforms?

**A**:
- **Same OS + CUDA**: Yes, fully compatible
- **Different OS**: May need platform-specific adjustments
- **Different CUDA**: Generate platform-specific lock file
- **CPU vs GPU**: May have different package versions

### Q: What if a package in lock file is deprecated?

**A**:
1. Find suitable replacement package
2. Update in source requirements (requirements/*.txt)
3. Regenerate lock files
4. Test thoroughly
5. Update documentation and migration guide
6. Communicate changes to team

### Q: Should I commit lock files to Git?

**A**: **Yes**, always commit lock files:
- Ensures team uses same versions
- Enables reproducible builds in CI/CD
- Provides audit trail
- Facilitates debugging

### Q: How do I handle merge conflicts in lock files?

**A**:
```bash
# Accept one version
git checkout --theirs requirements/lock/ml.lock

# Or regenerate
bash scripts/ci/update_lock_files.sh ml

# Always test after resolution
pip install -r requirements/lock/ml.lock
pytest tests/
```

### Q: What's the difference between lock file and requirements.txt?

**A**:

| Aspect | requirements.txt | lock file |
|--------|-----------------|-----------|
| Versions | Ranges (>=, <, ~=) | Exact (==) |
| Purpose | Specify dependencies | Freeze versions |
| Updates | Flexible | Fixed |
| Use case | Development | Production |
| Size | Smaller | Larger |
| Dependencies | Direct only | All transitive |

## Best Practices

### Development Workflow

```bash
# 1. Daily development - use source requirements
pip install -r requirements/ml.txt

# 2. Experiment freely with new packages
pip install experimental-package

# 3. Before committing - test with lock file
pip install -r requirements/lock/ml.lock
pytest tests/

# 4. Update lock file if requirements changed
bash scripts/ci/update_lock_files.sh ml

# 5. Commit both source and lock files
git add requirements/ml.txt requirements/lock/ml.lock
git commit -m "feat: add new dependency"
```

### Production Deployment

```bash
# 1. Always use lock files in production
pip install -r requirements/lock/all.lock

# 2. Verify installation
python scripts/setup/verify_installation.py

# 3. Run smoke tests
pytest tests/smoke/ -v

# 4. Monitor for issues
tail -f /var/log/application.log
```

### Team Collaboration

```bash
# 1. Pull latest changes
git pull origin main

# 2. Install from lock file
pip install -r requirements/lock/ml.lock

# 3. Report any installation issues
# Open GitHub issue with:
# - Lock file name
# - Error message
# - Platform details
# - Python version
```

### Research Reproducibility

```bash
# 1. Document exact versions in paper
cat requirements/lock/llm.lock | grep "^torch=="
cat requirements/lock/llm.lock | grep "^transformers=="

# 2. Include lock file hash in paper
sha256sum requirements/lock/llm.lock

# 3. Archive lock file with results
cp requirements/lock/llm.lock results/dependencies/
```

## Related Documentation

### Project Documentation
- **Installation Guide**: `docs/getting_started/installation.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`
- **Health Checks**: `HEALTH_CHECK.md`
- **Quick Start**: `QUICK_START.md`

### Requirements Files
- **Source Requirements**: `requirements/*.txt`
- **Compatibility Matrix**: `configs/compatibility_matrix.yaml`
- **Platform Guides**: `docs/platform_guides/`

### Scripts
- **Update Script**: `scripts/ci/update_lock_files.sh`
- **Verification**: `scripts/setup/verify_installation.py`
- **Health Check**: `src/core/health/health_checker.py`

### Guides
- **SOTA Models**: `SOTA_MODELS_GUIDE.md`
- **Platform Optimization**: `PLATFORM_OPTIMIZATION_GUIDE.md`
- **Free Deployment**: `FREE_DEPLOYMENT_GUIDE.md`

## Contributing

To contribute lock file updates or documentation:

### Update Process

1. **Create Issue**: Describe reason for update
2. **Fork Repository**: Work in your own fork
3. **Create Branch**: `update-dependencies-YYYY-MM`
4. **Update Files**: Run update scripts
5. **Test Thoroughly**: All tests must pass
6. **Security Audit**: No vulnerabilities
7. **Documentation**: Update CHANGELOG.md
8. **Pull Request**: Include all changes

### Pull Request Checklist

- [ ] Lock files regenerated successfully
- [ ] All tests pass with new versions
- [ ] Security audit completed (0 critical/high vulnerabilities)
- [ ] CHANGELOG.md updated
- [ ] Migration guide created (if breaking changes)
- [ ] Compatibility matrix updated
- [ ] Documentation reviewed

## Support and Resources

### Getting Help

For issues with lock files:

1. **Check This README**: Comprehensive troubleshooting section
2. **Search Issues**: Look for similar problems
3. **Check Troubleshooting Guide**: `TROUBLESHOOTING.md`
4. **Open New Issue**: Provide detailed information

### Issue Template

When opening an issue about lock files, include:

```markdown
**Lock File**: requirements/lock/ml.lock
**Python Version**: 3.10.12
**Platform**: Ubuntu 22.04 LTS
**CUDA**: 11.8

**Error Message**:
```
[Paste full error message]
```

**Steps to Reproduce**:
1. Create virtual environment
2. Run: pip install -r requirements/lock/ml.lock
3. Error occurs at package X

**Expected Behavior**: Installation completes successfully

**Additional Context**: [Any other relevant information]
```

## Acknowledgments

Lock file system design inspired by:
- Python pip-tools
- Poetry lock files
- Pipenv lock files
- Conda environment exports
- Docker layer caching
- Reproducible research best practices

### Contact

- **GitHub Issues**: Primary support channel
- **Email**: vohaidung.work@gmail.com (for security issues)
- **Repository**: https://github.com/VoHaiDung/ag-news-text-classification

## License

This documentation and all lock files are part of the AG News Text Classification project.

**Copyright** (c) 2025 Võ Hải Dũng

**Licensed** under the MIT License. See the [LICENSE](LICENSE) file in the project root for full license text.

---

**Last Updated**: 2025-09-19  
**Version**: 1.0.0  
**Maintained by**: Võ Hải Dũng
