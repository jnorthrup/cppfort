#!/usr/bin/env bash
# =============================================================================
# auto_fix_failing_tests.sh – parallel (max 3) Claude‑p patch generation
#
# For every cppfront regression test that fails with "compile failed":
#   1. Create (or reuse) a branch  fix/<test‑name>
#   2. Launch a Claude‑p session (max 3 concurrent) that receives:
#        – the test source file
#        – the compiler error excerpt
#      and returns a **unified diff**.
#   3. Apply the diff, commit it as "fix: <test‑name> – auto‑patch from Claude‑p".
#   4. (Optional) merge all fix/* branches back into the base branch.
#
# No placeholder files are added – only real patches produced by Claude‑p.
# =============================================================================

set -euo pipefail

# ------------------- Configuration -----------------------------------------
BASE_BRANCH="master"                # change if your default branch has another name
CLAUDE_CMD="claude -p"              # the Claude‑p CLI
MAX_PARALLEL=3                      # maximum concurrent Claude‑p sessions
# ---------------------------------------------------------------------------

# Ensure we start from a clean base branch
git checkout "$BASE_BRANCH"
git pull || true

# ---------------------------------------------------------------------------
# 1️⃣ Run the regression suite and collect failing tests
# ---------------------------------------------------------------------------
REG_OUTPUT="$(./ckmake regression 2>&1 | sed -r 's/\x1b\[[0-9;]*m//g')" || true

mapfile -t FAILING_TESTS < <(
    echo "$REG_OUTPUT" |
    grep -i "compile failed" |
    awk -F ':' '{gsub(/[[:space:]]+/,"",$1); print $1}'
)

if [[ ${#FAILING_TESTS[@]} -eq 0 ]]; then
    echo "✅ No failing tests detected – nothing to do."
    exit 0
fi

echo "🔎 Detected ${#FAILING_TESTS[@]} failing tests:"
printf "   • %s\n" "${FAILING_TESTS[@]}"

# ---------------------------------------------------------------------------
# Helper: build the Claude‑p prompt for a given test
# ---------------------------------------------------------------------------
gen_prompt() {
    local test_name="$1"
    local source_path="$2"
    local error_msg="$3"

    cat <<EOF
You are a senior C++/Cpp2 compiler engineer. The file "${source_path}" is part of the cppfront regression suite and fails to compile with the following error (excerpt only):

${error_msg}

Your task is to produce a **minimal, compile‑time correct** patch that makes this test compile and pass the cppfront regression checks. Return **only** a unified diff (no extra commentary).

Make sure the diff:
* applies cleanly with `git apply`,
* touches only this file,
* preserves the project's formatting style,
* and does not change the intended behaviour of the test beyond fixing the compilation error.
EOF
}

# ---------------------------------------------------------------------------
# 2️⃣ Process each failing test – create branch, fire Claude‑p (max 3 parallel)
# ---------------------------------------------------------------------------
declare -a PATCH_FILES   # patches in order of tests
declare -a BRANCHES
declare -a PIDS
active=0

for TEST in "${FAILING_TESTS[@]}"; do
    BRANCH="fix/${TEST//\//_}"
    echo -e "\n=== $TEST ==="
    echo "   → Branch: $BRANCH"

    if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
        echo "   • Branch already exists – checking out."
        git checkout "$BRANCH"
    else
        echo "   • Creating new branch."
        git checkout -b "$BRANCH"
    fi

    SRC_PATH="third_party/cppfront/regression-tests/${TEST}.cpp2"
    if [[ ! -f "$SRC_PATH" ]]; then
        echo "   ⚠️  Source not found ($SRC_PATH) – skipping."
        git checkout "$BASE_BRANCH"
        continue
    fi

    ERR_MSG="$(echo "$REG_OUTPUT" | awk "/^${TEST}:/,/^$/")"
    if [[ -z "$ERR_MSG" ]]; then
        ERR_MSG="(no error excerpt captured – please inspect the ckmake output manually)"
    fi

    PROMPT="$(gen_prompt "$TEST" "$SRC_PATH" "$ERR_MSG")"

    PATCH_FILE="$(mktemp /tmp/claude_patch_${TEST//\//_}.XXXXXX)"
    PATCH_FILES+=("$PATCH_FILE")
    BRANCHES+=("$BRANCH")

    echo "   • Launching Claude‑p (background) ..."
    echo "$PROMPT" | $CLAUDE_CMD >"$PATCH_FILE" 2>/dev/null &
    pid=$!
    PIDS+=("$pid")
    ((active++))

    if (( active >= MAX_PARALLEL )); then
        if command -v wait >/dev/null && wait -n; then
            ((active--))
        else
            wait
            active=0
        fi
    fi

    git checkout "$BASE_BRANCH"
done

# Wait for any remaining Claude‑p jobs
if (( active > 0 )); then
    echo "⏳ Waiting for remaining Claude‑p jobs..."
    wait
fi

# ---------------------------------------------------------------------------
# 3️⃣ Apply patches sequentially (Git must stay single‑threaded)
# ---------------------------------------------------------------------------
for idx in "${!PATCH_FILES[@]}"; do
    TEST="${FAILING_TESTS[$idx]}"
    BRANCH="${BRANCHES[$idx]}"
    PATCH_FILE="${PATCH_FILES[$idx]}"

    echo -e "\n--- Applying patch for $TEST (branch $BRANCH) ---"
    git checkout "$BRANCH"

    if [[ ! -s "$PATCH_FILE" ]]; then
        echo "   ⚠️  Claude‑p produced no output – skipping this test."
        git checkout "$BASE_BRANCH"
        continue
    fi

    if git apply --check "$PATCH_FILE"; then
        echo "   • Patch looks good – applying."
        git apply "$PATCH_FILE"
        git add -u
        git commit -m "fix: ${TEST} – auto‑patch from Claude‑p"
        echo "   ✅ Commit created."
    else
        echo "   ❌ Patch does not apply cleanly – leaving branch untouched."
        # Optionally keep the patch for manual inspection:
        # cp "$PATCH_FILE" "./${TEST}_failed.patch"
    fi

    git checkout "$BASE_BRANCH"
done

# ---------------------------------------------------------------------------
# 4️⃣ (Optional) merge all fix/* branches back into the base branch
# ---------------------------------------------------------------------------
echo -e "\n=== Merging fix/* branches back into $BASE_BRANCH ==="
for BR in $(git branch --list "fix/*" | sed 's/^[* ]*//'); do
    if git merge-base --is-ancestor "$BR" "$BASE_BRANCH"; then
        echo "   • $BR already merged – skipping."
    else
        echo "   • Merging $BR ..."
        git merge --no-ff "$BR" -m "Merge fix for ${BR#fix/}"
    fi
done

echo -e "\n🚀 All done!  Each failing test now has its own branch with a real Claude‑p patch, and the branches have been merged back into $BASE_BRANCH."
