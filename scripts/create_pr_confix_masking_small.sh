#!/usr/bin/env bash
# Script: create_pr_confix_masking_small.sh
# Purpose: create branch, add starter files (if present), commit and open a draft PR using gh
# Run from repo root: bash scripts/create_pr_confix_masking_small.sh

set -euo pipefail

BRANCH="feat/confix-masking/small"
PR_TITLE="feat: confix masking (starter skeleton)"
PR_BODY_FILE="docs/prs/feat-confix-masking-small.md"
LABELS="feature,needs-review"

echo "Creating branch ${BRANCH}"
git checkout -b "${BRANCH}"

# Stage known starter files (no-op if missing)
git add include/orbit/confix_masking.h src/confix_masking.cpp tests/confix_masking_test.cpp "${PR_BODY_FILE}" 2>/dev/null || true

if git diff --staged --quiet; then
  echo "No staged changes detected. Ensuring PR body file is staged."
  git add "${PR_BODY_FILE}"
fi

if git diff --staged --quiet; then
  echo "Nothing to commit. Skipping commit step."
else
  git commit -m "feat(confix): add confix masking skeleton (ConfixContext, OrbitPatternExt) and starter tests"
fi

echo "Pushing branch to origin"
git push -u origin "${BRANCH}"

echo "Creating draft PR with gh"
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI not found. Install GitHub CLI and authenticate (gh auth login) and re-run this script."
  exit 2
fi

gh pr create \
  --title "${PR_TITLE}" \
  --body-file "${PR_BODY_FILE}" \
  --label "${LABELS}" \
  --draft

echo "Opening PR in web UI"
gh pr view --web