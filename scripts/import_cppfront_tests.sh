#!/usr/bin/env bash
set -euo pipefail

# Import cppfront regression tests into corpus/inputs and bind SHA256 checksums
# Usage: ./scripts/import_cppfront_tests.sh [--clone]
# If --clone is passed, the script will (re)clone https://github.com/hsutter/cppfront into third_party/cppfront

REPO_DIR="third_party/cppfront"
CORPUS_DIR="corpus"
INPUTS_DIR="$CORPUS_DIR/inputs"
DB_FILE="$CORPUS_DIR/sha256_database.txt"

if [ "${1:-}" = "--clone" ]; then
  echo "Cloning cppfront into $REPO_DIR..."
  rm -rf "$REPO_DIR"
  git clone https://github.com/hsutter/cppfront "$REPO_DIR"
fi

if [ ! -d "$REPO_DIR/regression-tests" ]; then
  echo "ERROR: regression-tests not found in $REPO_DIR. Try running with --clone or place cppfront checkout at $REPO_DIR" >&2
  exit 1
fi

mkdir -p "$INPUTS_DIR"

echo "Copying tests to $INPUTS_DIR..."
rsync -a --delete --include='*/' --include='*.cpp2' --include='*.cpp' --exclude='*' "$REPO_DIR/regression-tests/" "$INPUTS_DIR/"

echo "Computing SHA256 checksums..."
tmpfile=$(mktemp)
find "$INPUTS_DIR" -type f -name '*.cpp2' -print0 | sort -z | while IFS= read -r -d '' f; do
  sha=$(shasum -a 256 "$f" | awk '{print $1}')
  rel=${f#${INPUTS_DIR}/}
  echo "$sha $rel" >> "$tmpfile"
done

mkdir -p "$CORPUS_DIR"
{
  echo "# Input file SHA256 checksums"
  echo "[inputs]"
  cat "$tmpfile" | sort
  echo
  echo "# Cppfront output SHA256 checksums"
  echo "[outputs]"
} > "$DB_FILE"

rm -f "$tmpfile"

echo "Wrote $DB_FILE with input checksums and empty outputs section."

echo "Done. To fill outputs, run cppfront on the inputs and update the [outputs] section accordingly."
