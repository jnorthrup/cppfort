#!/bin/bash

# install_stage1.sh - Install cpp2.h header for Stage 1 transpiler
# This script installs the cpp2.h header to a standard location

set -e

# Default installation prefix
PREFIX="/usr/local"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix=*)
            PREFIX="${1#*=}"
            shift
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--prefix=PREFIX]"
            echo ""
            echo "Install the cpp2.h header file to a standard location."
            echo ""
            echo "Options:"
            echo "  --prefix=PREFIX    Installation prefix (default: /usr/local)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we have the cpp2.h file
if [ ! -f "include/cpp2.h" ]; then
    echo "Error: include/cpp2.h not found"
    echo "Please run this script from the cppfort root directory"
    exit 1
fi

# Create target directory
INCLUDE_DIR="$PREFIX/include"
echo "Creating directory $INCLUDE_DIR (may require sudo)"
sudo mkdir -p "$INCLUDE_DIR"

# Install the header file
echo "Installing cpp2.h to $INCLUDE_DIR"
sudo cp include/cpp2.h "$INCLUDE_DIR/"

echo "Installation complete!"
echo "You can now use the Stage 1 transpiler without -I flags"