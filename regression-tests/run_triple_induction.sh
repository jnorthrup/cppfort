#!/usr/bin/env bash
# ----------------------------------------------------------------------
# regression-tests/run_triple_induction.sh
# ----------------------------------------------------------------------
# Triple Induction Test Framework: Stage 2 → Stage 1 → Stage 0
# Unified test harness demonstrating self-improving compilation pipeline
# ----------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Triple Induction Framework: Stage 2 → Stage 1 → Stage 0     ║"
echo "║  Self-Improving cpp2 Compilation Pipeline                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Track overall metrics
TOTAL_CYCLE_TIME_START=$(date +%s)

# Stage 2 → Stage 1: Attestation Validation
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ Phase 1: Stage 2 → Stage 1 (Attestation Feedback)           │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "Validating that Stage 1 transpilation produces attestable binaries..."
echo ""

STAGE2_START=$(date +%s)
if "${SCRIPT_DIR}/run_attestation_tests.sh"; then
    STAGE2_STATUS="✅ PASS"
    STAGE2_CODE=0
else
    STAGE2_STATUS="⚠️  NEEDS IMPROVEMENT"
    STAGE2_CODE=1
fi
STAGE2_END=$(date +%s)
STAGE2_TIME=$((STAGE2_END - STAGE2_START))

echo ""
echo "Phase 1 Result: ${STAGE2_STATUS} (${STAGE2_TIME}s)"
echo ""

# Stage 1 → Stage 0: Error Analysis
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ Phase 2: Stage 1 → Stage 0 (Error Analysis Feedback)        │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "Analyzing transpilation errors to guide Stage 0 improvements..."
echo ""

STAGE1_START=$(date +%s)
if "${SCRIPT_DIR}/run_error_analysis.sh"; then
    STAGE1_STATUS="✅ PASS"
    STAGE1_CODE=0
else
    STAGE1_STATUS="🔄 IMPROVEMENT NEEDED"
    STAGE1_CODE=1
fi
STAGE1_END=$(date +%s)
STAGE1_TIME=$((STAGE1_END - STAGE1_START))

echo ""
echo "Phase 2 Result: ${STAGE1_STATUS} (${STAGE1_TIME}s)"
echo ""

# Stage 0 → Stage 2: Roundtrip Validation
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ Phase 3: Stage 0 → Stage 2 (Roundtrip Validation)           │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "Testing bidirectional transformation and attestation..."
echo ""

STAGE0_START=$(date +%s)
# Run the existing inductive test suite
if "${SCRIPT_DIR}/run_tests.sh" >/dev/null 2>&1; then
    STAGE0_STATUS="✅ PASS"
    STAGE0_CODE=0
else
    STAGE0_STATUS="🔄 REFINEMENT NEEDED"
    STAGE0_CODE=1
fi
STAGE0_END=$(date +%s)
STAGE0_TIME=$((STAGE0_END - STAGE0_START))

echo "Testing Stage 0 roundtrip capabilities..."
echo "  - AST generation and emission"
echo "  - Bidirectional transformation"
echo "  - Semantic preservation"
echo ""
echo "Phase 3 Result: ${STAGE0_STATUS} (${STAGE0_TIME}s)"
echo ""

# Overall metrics
TOTAL_CYCLE_TIME_END=$(date +%s)
TOTAL_TIME=$((TOTAL_CYCLE_TIME_END - TOTAL_CYCLE_TIME_START))

# Final summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Triple Induction Summary                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Phase Results:"
echo "──────────────"
echo "  Phase 1 (Stage 2→1): ${STAGE2_STATUS} [${STAGE2_TIME}s]"
echo "  Phase 2 (Stage 1→0): ${STAGE1_STATUS} [${STAGE1_TIME}s]"
echo "  Phase 3 (Stage 0→2): ${STAGE0_STATUS} [${STAGE0_TIME}s]"
echo ""
echo "Total Cycle Time: ${TOTAL_TIME}s"
echo ""

# Determine overall status
TOTAL_ERRORS=$((STAGE2_CODE + STAGE1_CODE + STAGE0_CODE))

if [[ ${TOTAL_ERRORS} -eq 0 ]]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  ✅ ALL PHASES PASSED - TRIPLE INDUCTION WORKING!           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 The compilation pipeline is self-improving:"
    echo "   • Stage 2 validates Stage 1 outputs via attestation"
    echo "   • Stage 1 errors guide Stage 0 improvements"
    echo "   • Stage 0 roundtrips ensure semantic correctness"
    echo ""
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  🔄 INDUCTION IN PROGRESS - ${TOTAL_ERRORS} PHASES NEED WORK           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Feedback Loop Active:"
    echo ""

    if [[ ${STAGE2_CODE} -ne 0 ]]; then
        echo "  🔧 Stage 1 needs improvement for attestation consistency"
        echo "     → Review transpiler output generation"
        echo "     → Verify deterministic code emission"
        echo ""
    fi

    if [[ ${STAGE1_CODE} -ne 0 ]]; then
        echo "  🔧 Stage 0 needs improvement for error reduction"
        echo "     → Review error analysis priority queue"
        echo "     → Enhance AST generation and emission"
        echo ""
    fi

    if [[ ${STAGE0_CODE} -ne 0 ]]; then
        echo "  🔧 Stage 0 roundtrip needs refinement"
        echo "     → Improve bidirectional transformation"
        echo "     → Verify semantic preservation"
        echo ""
    fi

    echo "💡 Re-run this script after improvements to track progress"
    echo ""
    exit 1
fi