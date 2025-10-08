# AG News Text Classification - Lock Files

## Overview

This directory contains locked dependency files with exact package versions for reproducible installations across different environments.

## Purpose

Lock files ensure:
- **Reproducibility**: Same versions across team members and environments
- **Stability**: Prevent unexpected breaking changes from new package releases
- **Security**: Track exact versions for vulnerability audits
- **CI/CD**: Consistent builds in automated pipelines
- **Production**: Identical dependencies between development and production

## Available Lock Files

### Core Lock Files

| File | Description | Size | Use Case |
|------|-------------|------|----------|
| `base.lock` | Core dependencies only | ~2GB | Minimal setup, testing |
| `ml.lock` | ML training dependencies | ~8GB | Model training |
| `llm.lock` | LLM-specific dependencies | ~10GB | LLM fine-tuning |
| `all.lock` | Complete dependency set | ~15GB | Full development |

### Lock File Hierarchy

```
base.lock (50 packages)
  └── ml.lock (150+ packages)
        └── llm.lock (200+ packages)
              └── all.lock (400+ packages)
```

## When to Use Lock Files

### Use Lock Files When:
- Setting up CI/CD pipelines
- Deploying to production
- Reproducing research results
- Debugging environment-specific issues
- Installing on air-gapped systems
- Need exact reproducibility

### Use Regular Requirements When:
- Local development
- Experimenting with new packages
- Want latest security patches
- Testing compatibility with newer versions

## Installation

### Install from Lock File

```bash
# Install exact versions
pip install -r requirements/lock/base.lock

# Or for full ML stack
pip install -r requirements/lock/ml.lock

# Or for LLM support
pip install -r requirements/lock/llm.lock

# Or everything
pip install -r requirements/lock/all.lock
```

### Install from Regular Requirements (Latest Versions)

```bash
# Install latest compatible versions
pip install -r requirements/base.txt
pip install -r requirements/ml.txt
pip install -r requirements/llm.txt
pip install -r requirements/all_local.txt
```

## Generating Lock Files

### Manual Generation

```bash
# Create clean virtual environment
python -m venv lock_env
source lock_env/bin/activate  # or lock_env\Scripts\activate on Windows

# Install from requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements/base.txt

# Generate lock file
pip freeze > requirements/lock/base.lock

# Cleanup
deactivate
rm -rf lock_env
```

### Automated Generation

Use the provided script:

```bash
bash scripts/ci/update_lock_files.sh
```

This script:
1. Creates temporary virtual environments
2. Installs from each requirements file
3. Generates corresponding lock files
4. Runs security audits
5. Validates installations
6. Cleans up temporary files

## Lock File Format

Lock files contain:
- Exact package versions (e.g., `torch==2.1.2`)
- Platform-specific builds (e.g., `torch==2.1.2+cu118`)
- All transitive dependencies
- Auto-installed dependencies

Example:
```text
# Direct dependency
transformers==4.37.2

# Transitive dependencies (auto-installed by transformers)
tokenizers==0.15.1
safetensors==0.4.2
huggingface-hub==0.20.3
```

## Platform-Specific Considerations

### CUDA Versions

Lock files in this directory are for **CUDA 11.8**:
- `torch==2.1.2+cu118`
- `torchvision==0.16.2+cu118`
- `torchaudio==2.1.2+cu118`

For other CUDA versions:

```bash
# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

# Then install rest
pip install -r requirements/ml.txt  # Not the lock file!
```

### CPU-Only

```bash
# Install CPU version of PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cpu

# Then install rest
pip install -r requirements/ml.txt
```

### Apple Silicon (M1/M2)

```bash
# Default PyTorch has MPS support
pip install -r requirements/lock/ml.lock
# (MPS backend automatically used)
```

### Windows

Some packages in lock files are Linux-specific:
- `flash-attn`: Skip on Windows
- `deepspeed`: Skip on Windows
- `horovod`: Skip on Windows

For Windows:
```bash
# Use regular requirements instead
pip install -r requirements/ml.txt
```

## Verification

### Verify Installation

```bash
# Check installed packages match lock file
pip list --format=freeze > installed.txt
diff requirements/lock/base.lock installed.txt
```

### Run Health Checks

```bash
python src/core/health/health_checker.py
python scripts/setup/verify_installation.py
python scripts/setup/verify_dependencies.py
```

### Test Imports

```python
# Test critical imports
python -c "
import torch
import transformers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Security

### Security Audits

Lock files are regularly audited for vulnerabilities:

```bash
# Install audit tools
pip install pip-audit safety

# Audit lock file
pip-audit -r requirements/lock/all.lock
safety check -r requirements/lock/all.lock

# Or use GitHub Dependabot (automated)
```

### Last Audit
- Date: 2024-01-15
- Critical vulnerabilities: 0
- High vulnerabilities: 0
- Tools: pip-audit, safety, snyk

### Security Best Practices
- Update lock files monthly
- Review security advisories
- Test updates in staging first
- Keep audit logs
- Document breaking changes

## Maintenance

### Update Schedule

| Lock File | Update Frequency | Reason |
|-----------|------------------|--------|
| base.lock | Monthly | Security patches |
| ml.lock | Monthly | New features, fixes |
| llm.lock | Monthly | LLM library updates |
| all.lock | Monthly | Complete refresh |

### Update Process

1. **Preparation**
   ```bash
   # Backup current lock files
   cp requirements/lock/*.lock requirements/lock/backup/
   
   # Create update branch
   git checkout -b update-dependencies-2024-02
   ```

2. **Generate New Lock Files**
   ```bash
   bash scripts/ci/update_lock_files.sh
   ```

3. **Testing**
   ```bash
   # Install new versions
   pip install -r requirements/lock/all.lock
   
   # Run full test suite
   pytest tests/ -v
   
   # Run benchmarks
   python experiments/benchmarks/accuracy_benchmark.py
   
   # Security audit
   pip-audit -r requirements/lock/all.lock
   ```

4. **Documentation**
   ```text
   # Update CHANGELOG.md
   ## [1.1.0] - 2024-02-15
   ### Changed
   - Updated all dependencies to latest versions
   - PyTorch 2.1.2 -> 2.2.0
   - Transformers 4.37.2 -> 4.38.0
   
   ### Fixed
   - CVE-2024-XXXXX in package XYZ
   ```

5. **Commit and Release**
   ```bash
   git add requirements/lock/*.lock CHANGELOG.md
   git commit -m "chore: update dependencies to 2024-02"
   git push origin update-dependencies-2024-02
   # Create PR, review, merge
   git tag v1.1.0
   git push origin v1.1.0
   ```

### Breaking Changes

When lock files have breaking changes:

1. **Document in CHANGELOG.md**
   ```markdown
   ## [2.0.0] - 2024-03-01
   ### BREAKING CHANGES
   - Transformers 4.x -> 5.0: API changes in AutoModel
   - See migration guide: docs/migrations/v2.0.0.md
   ```

2. **Create Migration Guide**
   ```markdown
   # Migration Guide: v1.x to v2.0
   
   ## Transformers 5.0 Changes
   - `AutoModel.from_pretrained()` now requires `trust_remote_code=True`
   - Updated code:
     ```python
     model = AutoModel.from_pretrained(
         "model_name",
         trust_remote_code=True  # New parameter
     )
     ```
   ```

3. **Increment Major Version**
   - v1.9.0 -> v2.0.0

## Compatibility Matrix

Lock files tested on:

| Platform | Python | CUDA | Status |
|----------|--------|------|--------|
| Ubuntu 20.04 | 3.8 | 11.8 | Tested |
| Ubuntu 20.04 | 3.9 | 11.8 | Tested |
| Ubuntu 20.04 | 3.10 | 11.8 | Tested |
| Ubuntu 22.04 | 3.10 | 11.8 | Tested |
| Ubuntu 22.04 | 3.11 | 11.8 | Tested |
| Windows 10 | 3.10 | 11.8 | Partial |
| Windows 11 | 3.10 | 11.8 | Partial |
| macOS 12 | 3.10 | N/A | CPU only |
| macOS 13 | 3.10 | N/A | CPU+MPS |
| Google Colab | 3.10 | 11.8 | Tested |
| Kaggle | 3.10 | 11.8 | Tested |

See `configs/compatibility_matrix.yaml` for detailed compatibility information.

## Troubleshooting

### Issue: Package Conflict

```text
ERROR: package-a 1.0 has requirement package-b>=2.0,
but you have package-b 1.5
```

**Solution:**
1. Check if using correct lock file
2. Clear pip cache: `pip cache purge`
3. Use fresh virtual environment
4. Report issue on GitHub

### Issue: CUDA Version Mismatch

```text
RuntimeError: CUDA version mismatch
```

**Solution:**
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch
# See: https://pytorch.org/get-started/locally/
```

### Issue: Platform-Specific Package

```text
ERROR: Could not find a version that satisfies the requirement
flash-attn==2.4.2 (from versions: none)
```

**Solution:**
- flash-attn is Linux-only
- Skip it on Windows/macOS
- Use regular requirements instead of lock file

### Issue: Out of Disk Space

```text
ERROR: No space left on device
```

**Solution:**
```bash
# Check disk space
df -h

# Clean pip cache
pip cache purge

# Clean conda cache (if using conda)
conda clean --all

# Clean Docker (if installed)
docker system prune -a
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: pip-${{ hashFiles('requirements/lock/ml.lock') }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements/lock/ml.lock
      
      - name: Run tests
        run: |
          pytest tests/ -v
```

### Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy lock file
COPY requirements/lock/ml.lock .

# Install exact versions
RUN pip install --no-cache-dir -r ml.lock

# Copy application
COPY . .

CMD ["python", "train.py"]
```

## FAQ

### Q: Why use lock files instead of version ranges?

**A:** Lock files provide:
- Exact reproducibility
- Consistent CI/CD builds
- Security audit trail
- Production stability

### Q: How often should I update lock files?

**A:** 
- Security updates: Immediately
- Regular updates: Monthly
- Major versions: When needed (with testing)

### Q: Can I use lock files across different platforms?

**A:** 
- Same OS/CUDA: Yes
- Different OS: May need adjustments
- Different CUDA: Generate platform-specific lock file

### Q: What if a package in lock file is deprecated?

**A:** 
1. Find replacement package
2. Update requirements/*.txt
3. Regenerate lock files
4. Test thoroughly
5. Update documentation

### Q: How to handle transitive dependencies?

**A:** 
- Lock files include ALL dependencies
- No manual management needed
- pip resolves automatically

## Best Practices

### Development Workflow

```bash
# 1. Use regular requirements for development
pip install -r requirements/ml.txt

# 2. Install in editable mode
pip install -e .

# 3. Experiment freely

# 4. Before committing, test with lock file
pip install -r requirements/lock/ml.lock
pytest tests/
```

### Production Deployment

```bash
# Always use lock files in production
pip install -r requirements/lock/all.lock

# Verify installation
python scripts/setup/verify_installation.py

# Run smoke tests
pytest tests/smoke/ -v
```

### Team Collaboration

```bash
# Share lock files via Git
git add requirements/lock/*.lock
git commit -m "chore: update lock files"

# Team members install exact versions
pip install -r requirements/lock/ml.lock
```

## Related Files

- Regular requirements: `requirements/*.txt`
- Compatibility matrix: `configs/compatibility_matrix.yaml`
- Update script: `scripts/ci/update_lock_files.sh`
- Installation guide: `docs/getting_started/installation.md`
- Troubleshooting: `TROUBLESHOOTING.md`

## Contributing

To contribute lock file updates:

1. Create issue describing need for update
2. Run `bash scripts/ci/update_lock_files.sh`
3. Test thoroughly
4. Run security audit
5. Update CHANGELOG.md
6. Submit PR with:
   - Updated lock files
   - Test results
   - Security audit results
   - Migration guide (if breaking changes)

## Support

For lock file issues:
- Check this README
- See TROUBLESHOOTING.md
- Search existing GitHub issues
- Open new issue with:
  - Lock file name
  - Python version
  - Platform
  - Error message
  - Steps to reproduce

## License

This documentation and lock files are part of the AG News Text Classification project.

Copyright (c) 2025 Võ Hải Dũng

Licensed under the MIT License. See LICENSE file for details.
