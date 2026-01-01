#!/bin/bash
# Lock corpus hashes - verifies cppfront reference hasn't changed
# Run this before making any cppfort changes to ensure ground truth is stable

set -e

CORPUS_DIR="$1"
if [ -z "$CORPUS_DIR" ]; then
    CORPUS_DIR="corpus"
fi

HASH_DB="$CORPUS_DIR/sha256_database.txt"
LOCK_FILE="$CORPUS_DIR/.hash_lock"

echo "=== Corpus Hash Lock Verification ==="
echo ""

# Check if hash database exists
if [ ! -f "$HASH_DB" ]; then
    echo "Error: Hash database not found: $HASH_DB"
    exit 1
fi

# Calculate current hashes
CURRENT_HASHES=$(mktemp)
echo "[inputs]" > "$CURRENT_HASHES"

for f in "$CORPUS_DIR/inputs"/*.cpp2; do
    if [ -f "$f" ]; then
        hash=$(sha256sum "$f" | awk '{print $1}')
        basename=$(basename "$f")
        echo "$hash $basename" >> "$CURRENT_HASHES"
    fi
done

# Compare with locked hashes
if [ -f "$LOCK_FILE" ]; then
    echo "Checking against locked hashes..."
    if diff -q "$LOCK_FILE" "$CURRENT_HASHES" > /dev/null 2>&1; then
        echo "✓ Corpus hashes match lock file"
        rm -f "$CURRENT_HASHES"
        exit 0
    else
        echo "✗ HASH MISMATCH DETECTED!"
        echo ""
        echo "Differences:"
        diff "$LOCK_FILE" "$CURRENT_HASHES" | head -20
        rm -f "$CURRENT_HASHES"
        echo ""
        echo "CORPUS HAS CHANGED - Update lock file with:"
        echo "  cp $CURRENT_HASHES $LOCK_FILE"
        exit 1
    fi
else
    echo "No lock file found, creating one..."
    cp "$CURRENT_HASHES" "$LOCK_FILE"
    echo "✓ Lock file created: $LOCK_FILE"
    rm -f "$CURRENT_HASHES"
    echo ""
    echo "WARNING: This is the FIRST TIME locking the corpus."
    echo "Future runs will verify hashes haven't changed."
    exit 0
fi
