#!/usr/bin/env bash
set -euo pipefail

# Base branch to work from (adjust if repository uses a different name)
BASE_BRANCH="master"

# Ensure we are on the base branch and up‑to‑date
git checkout "$BASE_BRANCH"
git pull || true

# Run the regression suite and collect names of tests that failed with "compile failed"
# Expected output lines look like: <test-name>: compile failed
FAILURES=$(./ckmake regression 2>&1 | grep -i "compile failed" | awk -F ':' '{print $1}' | tr -d ' ')

if [[ -z "$FAILURES" ]]; then
  echo "✅ No failing tests detected – nothing to do."
  exit 0
fi

echo "🔎 Detected failing tests:"
printf "%s\n" $FAILURES

# Arrays to track what we created and merged for the summary
declare -a CREATED_BRANCHES
declare -a MERGED_BRANCHES

# Create a branch per failing test (if it does not already exist) and add a placeholder file
for TEST in $FAILURES; do
  # Sanitize branch name – replace any '/' with '_' to keep it a valid branch name
  BRANCH="fix/${TEST//\//_}"
  if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "Branch $BRANCH already exists – skipping creation."
  else
    git checkout -b "$BRANCH"
    PLACEHOLDER="scripts/TODO_${TEST//\//_}.md"
    if [[ ! -f "$PLACEHOLDER" ]]; then
      echo "# Placeholder for $TEST" > "$PLACEHOLDER"
      echo "This file documents the required fix for test \`$TEST\`." >> "$PLACEHOLDER"
    fi
    git add "$PLACEHOLDER"
    git commit -m "chore: placeholder for fixing $TEST"
    CREATED_BRANCHES+=("$BRANCH")
    git checkout "$BASE_BRANCH"
  fi
done

# Merge each fix/* branch back into the base branch using --no-ff, if not already merged
for BRANCH in $(git branch --list "fix/*" | sed 's/^[* ]*//'); do
  if git merge-base --is-ancestor "$BRANCH" "$BASE_BRANCH"; then
    echo "Branch $BRANCH already merged – skipping."
    continue
  fi
  git merge --no-ff "$BRANCH" -m "chore: merge $BRANCH"
  MERGED_BRANCHES+=("$BRANCH")
done

# Summary
echo "--- Summary ---"
if (( ${#CREATED_BRANCHES[@]} )); then
  echo "Created branches:"
  for b in "${CREATED_BRANCHES[@]}"; do echo "- $b"; done
else
  echo "No new branches were created."
fi
if (( ${#MERGED_BRANCHES[@]} )); then
  echo "Merged branches:"
  for b in "${MERGED_BRANCHES[@]}"; do echo "- $b"; done
else
  echo "No new merges performed."
fi
