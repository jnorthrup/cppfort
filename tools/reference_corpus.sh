#!/bin/bash
# Generate reference corpus using cppfront (lazy/idempotent)
# Only regenerates if source .cpp2 is newer than cached output
#
# Creates:
#   tests/reference/<name>.cpp       - cppfront C++1 output
#   tests/reference/<name>.ast       - Clang AST dump of the C++1
#   tests/reference/<name>.sha256    - Hash of source for cache validation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPPFRONT="/Users/jim/work/cppfront/build/cppfront"
CPP2UTIL="/Users/jim/work/cppfront/include/cpp2util.h"
TEST_DIR="$PROJECT_ROOT/tests/regression-tests"
REF_DIR="$PROJECT_ROOT/tests/reference"

mkdir -p "$REF_DIR"

# Check cppfront exists
if [ ! -x "$CPPFRONT" ]; then
    echo "ERROR: cppfront not found at $CPPFRONT"
    echo "Build it: cd /Users/jim/work/cppfront && g++-15 -std=c++20 -o build/cppfront source/cppfront.cpp"
    exit 1
fi

# Hash a file (portable)
hash_file() {
    shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
}

# Check if reference needs regeneration
needs_regen() {
    local src="$1"
    local name="$2"
    local ref_cpp="$REF_DIR/${name}.cpp"
    local ref_hash="$REF_DIR/${name}.sha256"
    
    # No reference exists
    [ ! -f "$ref_cpp" ] && return 0
    [ ! -f "$ref_hash" ] && return 0
    
    # Source changed
    local current_hash=$(hash_file "$src")
    local cached_hash=$(cat "$ref_hash" 2>/dev/null)
    [ "$current_hash" != "$cached_hash" ] && return 0
    
    # Reference is up to date
    return 1
}

# Generate reference for a single file
generate_reference() {
    local src="$1"
    local name=$(basename "$src" .cpp2)
    local ref_cpp="$REF_DIR/${name}.cpp"
    local ref_ast="$REF_DIR/${name}.ast"
    local ref_hash="$REF_DIR/${name}.sha256"
    local ref_err="$REF_DIR/${name}.err"
    
    # Skip import-faking tests (they need special handling)
    if grep -q "^import " "$src" 2>/dev/null; then
        # Check if it's a real import or just import std
        if grep -E "^import [^s]|^import std\." "$src" >/dev/null 2>&1; then
            echo "SKIP (import): $name"
            return 0
        fi
    fi
    
    echo -n "GEN: $name... "
    
    # cppfront outputs to current directory, so we need to cd there
    local src_dir=$(dirname "$src")
    local src_base=$(basename "$src")
    local gen_cpp="${src_dir}/${name}.cpp"
    
    # Run cppfront from source directory
    if (cd "$src_dir" && "$CPPFRONT" "$src_base" >/dev/null 2>"$ref_err"); then
        if [ -f "$gen_cpp" ]; then
            mv "$gen_cpp" "$ref_cpp"
            
            # Generate AST dump (best effort)
            clang++ -std=c++20 -I/Users/jim/work/cppfront/include \
                -Xclang -ast-dump -fsyntax-only "$ref_cpp" \
                >"$ref_ast" 2>/dev/null || true
            
            # Save hash
            hash_file "$src" > "$ref_hash"
            
            rm -f "$ref_err"
            echo "OK"
        else
            echo "FAIL (no output)"
        fi
    else
        # Keep error file for diagnosis
        echo "FAIL (cppfront error)"
    fi
}

# Main
echo "=========================================="
echo "Reference Corpus Generator (Lazy/Idempotent)"
echo "=========================================="
echo ""

GENERATED=0
SKIPPED=0
CACHED=0

for src in "$TEST_DIR"/*.cpp2; do
    [ -f "$src" ] || continue
    name=$(basename "$src" .cpp2)
    
    if needs_regen "$src" "$name"; then
        generate_reference "$src"
        ((GENERATED++))
    else
        ((CACHED++))
    fi
done

echo ""
echo "=========================================="
echo "Summary: Generated=$GENERATED, Cached=$CACHED"
echo "Reference dir: $REF_DIR"
echo "=========================================="
