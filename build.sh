#!/bin/bash

# Cpp2 Transpiler Build Script
# Supports both make and ninja build systems

set -e

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
GENERATOR=${GENERATOR:-Ninja}
BUILD_DIR=${BUILD_DIR:-build}
JOBS=${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_deps() {
    print_status "Checking dependencies..."

    if ! command -v cmake &> /dev/null; then
        print_error "CMake is required but not installed"
        exit 1
    fi

    if [ "$GENERATOR" = "Ninja" ] && ! command -v ninja &> /dev/null; then
        print_warning "Ninja not found, falling back to make"
        GENERATOR="Unix Makefiles"
    fi

    # Check for MLIR
    if ! pkg-config --exists mlir 2>/dev/null && ! cmake --find-package -DNAME=MLIR -DCOMPILER_ID=GNU -DLANGUAGE=CXX -MODE=EXIST 2>/dev/null; then
        print_warning "MLIR not found, some features may be disabled"
    fi
}

# Setup build directory
setup_build() {
    print_status "Setting up build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
}

# Configure
configure() {
    print_status "Configuring with generator: $GENERATOR"
    print_status "Build type: $BUILD_TYPE"

    local extra_opts=""

    if [ "$BUILD_TYPE" = "Debug" ]; then
        extra_opts="-DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON"
    else
        extra_opts="-DCMAKE_BUILD_TYPE=Release"
    fi

    if [ -n "$INSTALL_PREFIX" ]; then
        extra_opts="$extra_opts -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    fi

    cmake .. -G "$GENERATOR" $extra_opts
}

# Build
build() {
    print_status "Building with $JOBS jobs..."

    if [ "$GENERATOR" = "Ninja" ]; then
        ninja -j "$JOBS"
    else
        make -j "$JOBS"
    fi
}

# Install
install() {
    print_status "Installing to $INSTALL_PREFIX"

    if [ "$GENERATOR" = "Ninja" ]; then
        ninja install
    else
        make install
    fi
}

# Run tests
test() {
    print_status "Running tests..."

    if [ "$GENERATOR" = "Ninja" ]; then
        ninja test
    else
        make test
    fi
}

# Clean
clean() {
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
}

# Show help
show_help() {
    cat << EOF
Cpp2 Transpiler Build Script

Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE     Build type: Debug|Release (default: Release)
    -g, --gen GEN       Build generator: Ninja|Unix (default: Ninja)
    -j, --jobs NUM      Number of parallel jobs (default: system)
    -d, --dir DIR       Build directory (default: build)
    -p, --prefix PREFIX Install prefix (default: /usr/local)
    -c, --clean         Clean build directory
    -T, --test          Run tests after build
    -i, --install       Install after build
    -h, --help          Show this help

Examples:
    $0                    # Default release build with ninja
    $0 -t Debug          # Debug build
    $0 -g Unix           # Use make instead of ninja
    $0 -j 8             # Use 8 parallel jobs
    $0 -c                # Clean build directory
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -g|--gen)
            GENERATOR="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -d|--dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -c|--clean)
            clean
            exit 0
            ;;
        -T|--test)
            RUN_TESTS=1
            shift
            ;;
        -i|--install)
            DO_INSTALL=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
check_deps

# Save original directory
ORIG_DIR=$(pwd)
setup_build

# Run configure if not already configured
if [ ! -f "Makefile" ] && [ ! -f "build.ninja" ]; then
    configure
else
    print_status "Build already configured"
fi

# Build
build

# Run tests if requested
if [ -n "$RUN_TESTS" ]; then
    test
fi

# Install if requested
if [ -n "$DO_INSTALL" ]; then
    install
fi

print_status "Build complete!"
cd "$ORIG_DIR"