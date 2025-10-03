#!/bin/bash

# uninstall_stage1.sh - Uninstall cpp2.h header for Stage 1 transpiler
# This script removes the cpp2.h header from the standard location

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
            echo "Uninstall the cpp2.h header file from the standard location."
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

# Define the installed file
INSTALLED_FILE="$PREFIX/include/cpp2.h"

# Check if the file exists
if [ ! -f "$INSTALLED_FILE" ]; then
    echo "Warning: $INSTALLED_FILE not found"
    echo "Nothing to uninstall"
    exit 0
fi

# Remove the header file
echo "Removing $INSTALLED_FILE (may require sudo)"
sudo rm "$INSTALLED_FILE"

echo "Uninstallation complete!"