#!/bin/bash
# Wrapper to run inference tools with proper libclang path

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

# Find libclang
LIBCLANG_PATH=""
for candidate in \
    "/opt/homebrew/opt/llvm/lib/libclang.dylib" \
    "/opt/homebrew/Cellar/llvm/*/lib/libclang.dylib" \
    "/usr/local/opt/llvm/lib/libclang.dylib" \
    "/usr/lib/x86_64-linux-gnu/libclang-*.so.1" \
    "/usr/lib/libclang.so"
do
    if [ -e "$candidate" ] || compgen -G "$candidate" > /dev/null 2>&1; then
        LIBCLANG_PATH=$(compgen -G "$candidate" 2>/dev/null | head -1)
        [ -n "$LIBCLANG_PATH" ] && break
    fi
done

if [ -z "$LIBCLANG_PATH" ]; then
    echo "ERROR: libclang not found. Install with 'brew install llvm' on macOS or 'apt install libclang-dev' on Linux."
    exit 1
fi

echo "Using libclang: $LIBCLANG_PATH"
export LIBCLANG_PATH

# Ensure venv
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" 2>/dev/null || true
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: failed to initialize virtual environment at $VENV_DIR"
    exit 1
fi

# Activate and ensure deps
source "$VENV_DIR/bin/activate"
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Run the tool
exec python "$@"
