#!/bin/bash

"""
CUDA Installation Script for AG News Classification Framework
=============================================================

This script installs CUDA toolkit and configures GPU environment following:
- NVIDIA (2023): "CUDA Installation Guide for Linux"
- Paszke et al. (2019): "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
- Strubell et al. (2019): "Energy and Policy Considerations for Deep Learning in NLP"

Author: Võ Hải Dũng
License: MIT
"""

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly CUDA_VERSION="11.8"
readonly CUDNN_VERSION="8.6"
readonly NCCL_VERSION="2.15"
readonly TENSORRT_VERSION="8.5"

# Color codes for output (no emoji)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

log_header() {
    echo -e "\n${BLUE}=================================================================================${RESET}"
    echo -e "${BLUE}$1${RESET}"
    echo -e "${BLUE}=================================================================================${RESET}\n"
}

# Detect Linux distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

# Detect Ubuntu version
detect_ubuntu_version() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "$VERSION_ID"
    else
        echo "unknown"
    fi
}

# Check if running with sufficient privileges
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        log_info "Please run without sudo. The script will request sudo when needed."
        exit 1
    fi
    
    if ! sudo -n true 2>/dev/null; then
        log_info "This script requires sudo privileges for some operations."
        sudo -v
    fi
}

# Check system requirements
check_system_requirements() {
    log_header "Checking System Requirements"
    
    # Check OS
    local distro=$(detect_distro)
    if [[ "$distro" != "ubuntu" && "$distro" != "debian" ]]; then
        log_warning "This script is designed for Ubuntu/Debian. Your OS: $distro"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        log_error "CUDA requires x86_64 architecture. Your architecture: $arch"
        exit 1
    fi
    
    # Check kernel version
    local kernel_version=$(uname -r)
    log_info "Kernel version: $kernel_version"
    
    # Check available disk space
    local available_space=$(df /usr/local | awk 'NR==2 {print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log_warning "Insufficient disk space. At least 10GB required in /usr/local"
        log_warning "Available: $((available_space / 1024 / 1024))GB"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Check for NVIDIA GPU
check_nvidia_gpu() {
    log_header "Checking for NVIDIA GPU"
    
    # Check if NVIDIA GPU is present
    if ! lspci | grep -i nvidia > /dev/null; then
        log_error "No NVIDIA GPU detected"
        log_info "CUDA installation requires an NVIDIA GPU"
        exit 1
    fi
    
    # Get GPU information
    local gpu_info=$(lspci | grep -i nvidia)
    log_info "Detected GPU:"
    echo "$gpu_info"
    
    # Check compute capability
    log_info "Checking GPU compute capability..."
    
    # This requires nvidia-smi to be installed
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,compute_cap --format=csv
    else:
        log_warning "nvidia-smi not found. Will install with CUDA toolkit."
    fi
    
    log_success "NVIDIA GPU detected"
}

# Remove old CUDA installations
remove_old_cuda() {
    log_header "Removing Old CUDA Installations"
    
    log_info "Checking for existing CUDA installations..."
    
    # Check common CUDA locations
    local cuda_dirs=(
        "/usr/local/cuda"
        "/usr/local/cuda-*"
        "/opt/cuda"
    )
    
    local found_cuda=false
    for dir in "${cuda_dirs[@]}"; do
        if ls $dir 2>/dev/null; then
            found_cuda=true
            log_warning "Found existing CUDA installation: $dir"
        fi
    done
    
    if [[ "$found_cuda" == true ]]; then
        read -p "Remove existing CUDA installations? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Remove CUDA packages
            log_info "Removing CUDA packages..."
            sudo apt-get remove --purge -y cuda* nvidia-cuda-* || true
            sudo apt-get remove --purge -y libcudnn* || true
            
            # Remove CUDA directories
            sudo rm -rf /usr/local/cuda*
            
            log_success "Old CUDA installations removed"
        fi
    else
        log_info "No existing CUDA installations found"
    fi
}

# Install NVIDIA drivers
install_nvidia_drivers() {
    log_header "Installing NVIDIA Drivers"
    
    # Check if drivers are already installed
    if nvidia-smi &> /dev/null; then
        local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        log_info "NVIDIA driver already installed: version $driver_version"
        
        # Check if driver version is compatible with CUDA 11.8
        local min_driver_version="520.61.05"
        if [[ "$driver_version" < "$min_driver_version" ]]; then
            log_warning "Driver version $driver_version is older than recommended $min_driver_version"
            log_info "Consider updating your NVIDIA drivers"
        else
            log_success "NVIDIA driver version is compatible"
            return 0
        fi
    fi
    
    log_info "Installing NVIDIA drivers..."
    
    # Add NVIDIA PPA
    sudo add-apt-repository ppa:graphics-drivers/ppa -y
    sudo apt-get update
    
    # Install recommended driver
    log_info "Installing recommended NVIDIA driver..."
    sudo ubuntu-drivers autoinstall
    
    log_success "NVIDIA drivers installed"
    log_warning "System reboot required for driver activation"
}

# Install CUDA toolkit
install_cuda_toolkit() {
    log_header "Installing CUDA Toolkit ${CUDA_VERSION}"
    
    local ubuntu_version=$(detect_ubuntu_version)
    local cuda_repo_pkg=""
    
    # Determine correct repo package based on Ubuntu version
    case "$ubuntu_version" in
        "22.04")
            cuda_repo_pkg="cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb"
            ;;
        "20.04")
            cuda_repo_pkg="cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb"
            ;;
        "18.04")
            cuda_repo_pkg="cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb"
            ;;
        *)
            log_error "Unsupported Ubuntu version: $ubuntu_version"
            exit 1
            ;;
    esac
    
    # Download CUDA installer
    log_info "Downloading CUDA toolkit..."
    local cuda_url="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/${cuda_repo_pkg}"
    local cuda_installer="/tmp/${cuda_repo_pkg}"
    
    if [[ ! -f "$cuda_installer" ]]; then
        wget -O "$cuda_installer" "$cuda_url" || {
            log_error "Failed to download CUDA installer"
            log_info "Please download manually from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
            exit 1
        }
    fi
    
    # Install CUDA repository
    log_info "Installing CUDA repository..."
    sudo dpkg -i "$cuda_installer"
    sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    
    # Install CUDA toolkit
    log_info "Installing CUDA toolkit..."
    sudo apt-get install -y cuda-toolkit-11-8
    
    # Install additional CUDA libraries
    log_info "Installing additional CUDA libraries..."
    sudo apt-get install -y \
        cuda-drivers \
        cuda-runtime-11-8 \
        cuda-compiler-11-8 \
        cuda-libraries-11-8 \
        cuda-libraries-dev-11-8
    
    log_success "CUDA toolkit ${CUDA_VERSION} installed"
}

# Install cuDNN
install_cudnn() {
    log_header "Installing cuDNN ${CUDNN_VERSION}"
    
    log_info "Installing cuDNN..."
    
    # Install from Ubuntu repository (easier than manual download)
    sudo apt-get install -y libcudnn8=${CUDNN_VERSION}* libcudnn8-dev=${CUDNN_VERSION}*
    
    # Hold packages to prevent automatic updates
    sudo apt-mark hold libcudnn8 libcudnn8-dev
    
    log_success "cuDNN ${CUDNN_VERSION} installed"
}

# Install NCCL
install_nccl() {
    log_header "Installing NCCL ${NCCL_VERSION}"
    
    log_info "Installing NCCL for multi-GPU support..."
    
    # Install NCCL
    sudo apt-get install -y libnccl2 libnccl-dev
    
    log_success "NCCL installed"
}

# Configure environment variables
configure_environment() {
    log_header "Configuring Environment Variables"
    
    log_info "Setting up CUDA environment variables..."
    
    # Create environment setup script
    local env_script="$PROJECT_ROOT/scripts/setup/cuda_env.sh"
    
    cat > "$env_script" << 'EOF'
#!/bin/bash
# CUDA Environment Configuration
# Generated by install_cuda.sh

# CUDA paths
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# CUDA device configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# PyTorch CUDA settings
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

# TensorRT paths (if installed)
if [[ -d "/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
fi

# Verify CUDA installation
if command -v nvcc &> /dev/null; then
    echo "CUDA $(nvcc --version | grep release | awk '{print $6}' | cut -c2-) available"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
fi
EOF
    
    chmod +x "$env_script"
    
    # Add to .bashrc if not already present
    if ! grep -q "cuda_env.sh" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA environment" >> ~/.bashrc
        echo "if [[ -f $env_script ]]; then" >> ~/.bashrc
        echo "    source $env_script" >> ~/.bashrc
        echo "fi" >> ~/.bashrc
    fi
    
    # Update .env file
    local env_file="$PROJECT_ROOT/.env"
    if [[ -f "$env_file" ]]; then
        log_info "Updating .env file..."
        
        # Remove old CUDA settings
        sed -i '/^CUDA_/d' "$env_file"
        
        # Add new CUDA settings
        cat >> "$env_file" << EOF

# CUDA Configuration (updated by install_cuda.sh)
CUDA_HOME=/usr/local/cuda-11.8
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0
EOF
    fi
    
    log_success "Environment variables configured"
    log_info "Run 'source ~/.bashrc' to activate CUDA environment"
}

# Verify CUDA installation
verify_cuda_installation() {
    log_header "Verifying CUDA Installation"
    
    # Source environment
    source "$PROJECT_ROOT/scripts/setup/cuda_env.sh"
    
    # Check nvcc
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
        log_success "CUDA compiler (nvcc) version: $cuda_version"
    else
        log_error "nvcc not found in PATH"
        return 1
    fi
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        log_success "nvidia-smi available"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        log_error "nvidia-smi not found"
        return 1
    fi
    
    # Test CUDA with simple program
    log_info "Testing CUDA with simple program..."
    
    local test_file="/tmp/cuda_test.cu"
    cat > "$test_file" << 'EOF'
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
    
    if nvcc -o /tmp/cuda_test "$test_file" 2>/dev/null; then
        if /tmp/cuda_test 2>/dev/null | grep -q "Hello from GPU"; then
            log_success "CUDA test program executed successfully"
        else
            log_warning "CUDA compilation succeeded but execution failed"
        fi
        rm -f /tmp/cuda_test
    else
        log_warning "CUDA test compilation failed"
    fi
    
    rm -f "$test_file"
    
    # Test PyTorch CUDA support
    log_info "Testing PyTorch CUDA support..."
    
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
" || log_warning "PyTorch CUDA test failed. You may need to reinstall PyTorch with CUDA support."
    
    log_success "CUDA installation verification complete"
}

# Install PyTorch with CUDA support
install_pytorch_cuda() {
    log_header "Installing PyTorch with CUDA Support"
    
    log_info "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
    
    # Activate virtual environment if it exists
    if [[ -d "$PROJECT_ROOT/venv" ]]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    fi
    
    # Install PyTorch with CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    log_success "PyTorch with CUDA support installed"
}

# Main installation flow
main() {
    log_header "CUDA Installation for AG News Classification"
    
    echo "This script will install CUDA ${CUDA_VERSION} and configure GPU support."
    echo "Following best practices from:"
    echo "  - NVIDIA CUDA Installation Guide"
    echo "  - PyTorch GPU Setup Documentation"
    echo ""
    
    # Check prerequisites
    check_privileges
    check_system_requirements
    check_nvidia_gpu
    
    # Installation steps
    remove_old_cuda
    install_nvidia_drivers
    install_cuda_toolkit
    install_cudnn
    install_nccl
    configure_environment
    verify_cuda_installation
    install_pytorch_cuda
    
    # Final instructions
    log_header "Installation Complete"
    
    echo -e "${GREEN}CUDA ${CUDA_VERSION} has been successfully installed!${RESET}"
    echo ""
    echo "Next steps:"
    echo "  1. Reboot your system if you installed new NVIDIA drivers"
    echo "  2. Run: source ~/.bashrc"
    echo "  3. Verify with: nvidia-smi"
    echo "  4. Test PyTorch: python -c 'import torch; print(torch.cuda.is_available())'"
    echo ""
    echo "For multi-GPU training, see documentation on distributed training."
    
    log_success "Setup complete!"
}

# Run main function
main "$@"
