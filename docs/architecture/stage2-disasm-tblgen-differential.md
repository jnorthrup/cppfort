# Stage2: Disasm → TableGen Pattern Extraction with Differential Tracking

**Status:** Late Stage2 goal | **Foundation Required:** -Og baseline at regression-time

## Problem Statement

Sea of Nodes optimizations eliminate node subgraphs entirely. At -O3, patterns visible at -Og may be "clipped to nothing" by:

- Constant folding (AddNode(2, 3) → ConstantNode(5))
- Dead code elimination (unused result chains)
- Common subexpression elimination (duplicate subgraphs merged)
- Loop unrolling (LoopNode → unrolled straight-line code)

**Challenge:** How do we extract TableGen patterns when the IR patterns disappear during optimization?

**Solution:** Differential tracking captures optimization deltas across -O levels.

---

## Architecture

```
CPP2 Source
    ↓ compile @ -Og, -O1, -O2, -O3
Binaries @ Each Opt Level
    ↓ disasm
Assembly Patterns @ Each Opt Level
    ↓ diff tracking
Differential Pattern Database
    ↓ induction
TableGen Pattern Candidates
    ↓ validation
Canonical TableGen Definitions
```

### Components

**1. Regression-Time -Og Baseline**

```bash
# Capture baseline patterns at -Og (minimal optimization)
./build/stage1_cli -Og example.cpp2 -o example_Og.out
objdump -d example_Og.out > example_Og.asm
```

**-Og preserves SON structure in assembly:**

- AddNode → visible `add` instruction
- MulNode → visible `mul` instruction
- LoopNode → visible loop structure

**2. Differential Extraction**

```bash
# Compile at all opt levels
for opt in Og O1 O2 O3; do
    ./build/stage1_cli -$opt example.cpp2 -o example_$opt.out

    # Use Ghidra for better basic block extraction (preferred)
    analyzeHeadless /tmp ghidra_project_${opt} \
        -import example_$opt.out \
        -postScript extract_cfg.py \
        -scriptPath ./scripts/ghidra

    # Fallback: objdump for simple disassembly
    objdump -d example_$opt.out > example_$opt.asm
done

# Extract differential using Ghidra CFG data
./scripts/extract_pattern_differential.py \
    --baseline ghidra_project_Og/cfg.json \
    --optimized ghidra_project_O1/cfg.json \
                ghidra_project_O2/cfg.json \
                ghidra_project_O3/cfg.json \
    --output pattern_diff.json
```

**3. Stochastic Pattern Tracking**

Stochastic cache items = patterns that appear/disappear based on optimization context.

```json
{
  "pattern_id": "add_const_fold",
  "source_pattern": "AddNode(ConstantNode, ConstantNode)",
  "idempotent": true,
  "recompute_cost": 1,
  "storage_policy": "recompute_on_demand",
  "appearances": {
    "Og": {"present": true, "asm": "mov eax, 2; add eax, 3"},
    "O1": {"present": false, "asm": null, "reason": "constant_folded"},
    "O2": {"present": false, "asm": null, "reason": "constant_folded"},
    "O3": {"present": false, "asm": null, "reason": "constant_folded"}
  },
  "differential": {
    "Og→O1": {"cost_reduction": 2, "eliminated": true},
    "O1→O2": {"cost_reduction": 0, "eliminated": false},
    "O2→O3": {"cost_reduction": 0, "eliminated": false}
  }
}
```

**4. Differential Equation Integration**

Connects to rainbow table model: `dC/dO = -α*C`

```python
def compute_pattern_differential(pattern, opt_levels):
    """Track cost reduction across optimization levels."""
    costs = {}
    for opt in opt_levels:
        asm = extract_asm_for_pattern(pattern, opt)
        if asm is None:
            costs[opt] = 0  # Pattern eliminated
        else:
            costs[opt] = len(asm.instructions)

    # Fit differential equation
    alpha = fit_exponential_decay(costs)

    # Determine storage policy (idempotent optimization)
    is_idempotent = pattern.is_pure_function()
    recompute_cost = pattern.estimate_recompute_cost()
    storage_policy = 'store' if recompute_cost > RECOMPUTE_THRESHOLD else 'recompute_on_demand'

    return {
        'costs': costs,
        'alpha': alpha,
        'model': f'C(O) = {costs["Og"]} * e^(-{alpha}*O)',
        'idempotent': is_idempotent,
        'recompute_cost': recompute_cost,
        'storage_policy': storage_policy
    }
```

---

## Stochastic Pattern Categories

### 1. Always Preserved

Patterns that survive all optimization levels:

```cpp
// External function call - cannot be eliminated
extern int foo();
int bar() { return foo(); }  // Preserved at all -O levels
```

**TableGen extraction:** Stable baseline patterns, extract at -Og.

### 2. Conditionally Eliminated

Patterns eliminated only when optimization conditions met:

```cpp
int add(int a, int b) { return a + b; }
int main() {
    return add(2, 3);  // Eliminated at -O1+ (inlined + const folded)
}
```

**TableGen extraction:** Capture at -Og, track elimination threshold.

### 3. Stochastically Appears

Patterns that only appear at higher optimization (new instructions):

```cpp
for (int i = 0; i < 100; i++) {
    arr[i] = i * 2;
}
// -Og: scalar loop
// -O3: vectorized (vpmulld xmm0, xmm1, xmm2) - NEW pattern
```

**TableGen extraction:** Compare -Og vs -O3, extract vectorized patterns.

### 4. Completely Eliminated

Patterns that SON clips to nothing:

```cpp
int unused() { return 2 + 3; }  // Dead code, eliminated everywhere
```

**TableGen extraction:** Skip (no assembly footprint).

### 5. Idempotent Recomputable

Patterns with deterministic results that are cheap to recompute:

```cpp
AddNode(ConstantNode(2), ConstantNode(3))  → ConstantNode(5)
// Cost: single integer addition (1 cycle)
// Storage: unnecessary (recompute on-demand)
```

**TableGen extraction:** Mark as `recompute_on_demand`, don't store unless cost > threshold.

**See:** [Idempotent Pattern Optimization](idempotent-pattern-optimization.md) for full storage policy analysis.

---

## Differential Tracking Algorithm

```python
class DifferentialPatternExtractor:
    def __init__(self, baseline_asm, opt_levels):
        self.baseline = parse_asm(baseline_asm)
        self.opt_asm = {o: parse_asm(asm) for o, asm in opt_levels.items()}
        self.patterns = {}

    def extract_differentials(self):
        # Step 1: Identify baseline patterns
        baseline_patterns = self.identify_patterns(self.baseline)

        for pattern in baseline_patterns:
            diff = {
                'pattern_id': pattern.id,
                'source_ir': pattern.source_node,
                'appearances': {},
                'differential': {}
            }

            # Step 2: Track pattern across opt levels
            for opt in ['Og', 'O1', 'O2', 'O3']:
                asm = self.opt_asm[opt]
                match = self.find_pattern(pattern, asm)

                if match:
                    diff['appearances'][opt] = {
                        'present': True,
                        'asm': match.asm_code,
                        'cost': len(match.instructions)
                    }
                else:
                    diff['appearances'][opt] = {
                        'present': False,
                        'asm': None,
                        'reason': self.infer_elimination_reason(pattern, opt)
                    }

            # Step 3: Compute differential costs
            costs = [diff['appearances'][o].get('cost', 0)
                     for o in ['Og', 'O1', 'O2', 'O3']]

            for i in range(len(costs) - 1):
                opt_from = ['Og', 'O1', 'O2', 'O3'][i]
                opt_to = ['Og', 'O1', 'O2', 'O3'][i+1]
                diff['differential'][f'{opt_from}→{opt_to}'] = {
                    'cost_reduction': costs[i] - costs[i+1],
                    'eliminated': costs[i+1] == 0
                }

            self.patterns[pattern.id] = diff

        return self.patterns

    def infer_elimination_reason(self, pattern, opt_level):
        """Infer why pattern was eliminated."""
        # Heuristics based on pattern type and opt level
        if pattern.is_constant_arithmetic():
            return 'constant_folded'
        elif pattern.is_pure_function() and opt_level >= 'O2':
            return 'inlined_and_folded'
        elif pattern.is_dead_code():
            return 'dead_code_eliminated'
        else:
            return 'unknown_optimization'
```

---

## TableGen Pattern Database Generation

### Phase 1: Collect Baseline (-Og)

```bash
# Regression harness captures -Og baseline for all tests
for test in regression-tests/*.cpp2; do
    ./build/stage1_cli -Og $test -o ${test%.cpp2}_Og.out
    objdump -d ${test%.cpp2}_Og.out > ${test%.cpp2}_Og.asm
done

# Extract SON IR patterns
./scripts/extract_son_patterns.py \
    --ir-dump regression-tests/*.cpp2.ir \
    --output baseline_patterns.json
```

**baseline_patterns.json:**

```json
[
  {
    "pattern_id": "add_i32",
    "son_ir": "AddNode(type=i32, lhs=$a, rhs=$b)",
    "frequency": 1247,
    "test_files": ["test_arith.cpp2", "test_expr.cpp2"]
  },
  {
    "pattern_id": "mul_i32_const",
    "son_ir": "MulNode(type=i32, lhs=$a, rhs=ConstantNode($c))",
    "frequency": 583,
    "test_files": ["test_arith.cpp2"]
  }
]
```

### Phase 2: Track Differentials

```python
# Generate differential tracking data
./scripts/generate_differentials.py \
    --baseline baseline_patterns.json \
    --opt-levels Og O1 O2 O3 \
    --output pattern_differentials.json

# Example output
{
  "add_i32": {
    "baseline_cost": 2,
    "costs": {"Og": 2, "O1": 2, "O2": 1, "O3": 1},
    "alpha": 0.15,
    "survival_rate": 1.0,  # Appears in 100% of opt levels
    "elimination_threshold": null
  },
  "add_const_fold": {
    "baseline_cost": 3,
    "costs": {"Og": 3, "O1": 0, "O2": 0, "O3": 0},
    "alpha": 1.2,  # Aggressive elimination
    "survival_rate": 0.25,  # Appears in 25% of opt levels
    "elimination_threshold": "O1"
  }
}
```

### Phase 3: Generate TableGen Candidates

```python
# Convert stable patterns to TableGen definitions
./scripts/generate_tblgen_patterns.py \
    --differentials pattern_differentials.json \
    --min-survival-rate 0.5 \
    --output patterns.td

# patterns.td
def SON_Add_i32 : Pat<
    (SON_AddNode i32:$a, i32:$b),
    (MLIR_arith_AddIOp $a, $b)
>;

# Note: add_const_fold NOT generated (survival_rate < 0.5)
# It's a stochastic pattern eliminated at -O1+
```

---

## Integration with Rainbow Table

Rainbow table dispatch depends on baseline patterns:

```cpp
// Rainbow table entry
struct RainbowEntry {
    NodeKind pattern;        // From baseline -Og IR
    OptLevel opt_level;
    LoweringStrategy lower;
    uint32_t cost;           // From differential tracking
    bool stochastic;         // True if pattern may be eliminated
};

// Example entries
{NodeKind::ADD, OptLevel::Og, Lowering::Simple, cost: 2, stochastic: false},
{NodeKind::ADD, OptLevel::O3, Lowering::LEA,    cost: 1, stochastic: false},
{NodeKind::ADD_CONST, OptLevel::Og, Lowering::Simple, cost: 3, stochastic: true},
{NodeKind::ADD_CONST, OptLevel::O1, Lowering::Fold,   cost: 0, stochastic: true},
```

**Stochastic flag** indicates pattern may not appear in final assembly.

---

## Regression Test Integration

### CMakeLists.txt

```cmake
# Regression tests capture -Og baseline
add_custom_target(regression_baseline
    COMMAND ${CMAKE_SOURCE_DIR}/regression-tests/run_with_baseline.sh
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Extract differentials after baseline
add_custom_target(pattern_differentials
    DEPENDS regression_baseline
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/generate_differentials.py
        --baseline ${CMAKE_BINARY_DIR}/baseline_patterns.json
        --opt-levels Og O1 O2 O3
        --output ${CMAKE_BINARY_DIR}/pattern_differentials.json
)

# Generate TableGen patterns
add_custom_target(tblgen_patterns
    DEPENDS pattern_differentials
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/generate_tblgen_patterns.py
        --differentials ${CMAKE_BINARY_DIR}/pattern_differentials.json
        --output ${CMAKE_SOURCE_DIR}/patterns.td
)
```

### run_with_baseline.sh

```bash
#!/bin/bash
set -e

OPT_LEVELS=(Og O1 O2 O3)

for test in regression-tests/*.cpp2; do
    base=$(basename $test .cpp2)

    # Compile at all opt levels
    for opt in "${OPT_LEVELS[@]}"; do
        echo "Compiling $base at -$opt..."
        ./build/stage1_cli -$opt $test -o build/${base}_${opt}.out
        objdump -d build/${base}_${opt}.out > build/${base}_${opt}.asm
    done

    # Dump IR at -Og for baseline
    ./build/stage1_cli -Og --dump-ir $test > build/${base}.ir
done

# Extract baseline patterns
./scripts/extract_son_patterns.py \
    --ir-dump build/*.ir \
    --output build/baseline_patterns.json

echo "Baseline patterns captured: $(jq length build/baseline_patterns.json)"
```

---

## Validation Strategy

### 1. Differential Consistency Check

Ensure cost monotonically decreases (or stays flat):

```python
def validate_differential(pattern):
    costs = pattern['costs']
    for i in range(len(costs) - 1):
        assert costs[i] >= costs[i+1], \
            f"Cost increased from {costs[i]} to {costs[i+1]}"
```

### 2. Stochastic Pattern Detection

Flag patterns with high elimination variance:

```python
def detect_stochastic(pattern):
    survival_rate = sum(1 for c in pattern['costs'].values() if c > 0) / len(pattern['costs'])
    return survival_rate < 0.8  # Eliminated in >20% of opt levels
```

### 3. TableGen Pattern Validation

Only generate patterns with high survival rates:

```python
def should_generate_tblgen(pattern):
    return pattern['survival_rate'] >= 0.5  # Present in ≥50% of opt levels
```

---

## Benefits

### 1. Captures SON Optimization Effectiveness

Differential tracking quantifies what SON eliminates:

```
AddNode(Const, Const): 100% elimination at -O1+ (constant folding)
LoadNode(invariant):   80% elimination at -O2+ (hoisting)
MulNode($x, 2):        50% elimination at -O2+ (shift conversion)
```

**Ghidra advantage:** Basic block count reduction directly measures control flow simplification:
```
-Og: 47 basic blocks (unoptimized CFG)
-O3: 12 basic blocks (inlined, straightened, merged)
Reduction: 74% basic block elimination
```

### 2. Validates Optimization Correctness

If pattern eliminated at -O3 but binary still produces correct output → optimization valid.

### 3. Builds Comprehensive Pattern Database

Captures patterns that:

- Always appear (stable TableGen candidates)
- Conditionally appear (context-dependent patterns)
- Never appear in optimized code (training negatives for ML)

### 4. Feeds Autoencoder Training

Differential data provides labeled examples:

```python
# Training data for autoencoder
training_examples = [
    {'pattern': 'AddNode(Const, Const)', 'opt_level': 'Og', 'cost': 3, 'label': 'foldable'},
    {'pattern': 'AddNode(Const, Const)', 'opt_level': 'O1', 'cost': 0, 'label': 'folded'},
    {'pattern': 'LoadNode(invariant)', 'opt_level': 'Og', 'cost': 2, 'label': 'hoistable'},
    {'pattern': 'LoadNode(invariant)', 'opt_level': 'O2', 'cost': 1, 'label': 'hoisted'},
]
```

---

## Timeline

### Phase 1: Foundation (Band 5 completion)

- Implement -Og baseline capture in regression harness
- Add objdump disassembly step
- Create baseline pattern database

### Phase 2: Differential Tracking (Post-Band 5)

- Implement differential extraction script
- Add stochastic pattern detection
- Validate differential consistency

### Phase 3: TableGen Generation (Late Stage2)

- Build pattern → TableGen converter
- Generate patterns.td from differentials
- Validate against MLIR lowering

### Phase 4: ML Integration (Future)

- Feed differentials to autoencoder
- Train on optimization effectiveness
- Predict stochastic pattern behavior

---

## Tooling

**Ghidra headless:** Use for CFG/basic block extraction whenever we need better basic blocks than objdump provides.

**Ghidra advantages over objdump:**
- **Basic block boundaries** - Accurate CFG extraction (objdump: linear disassembly)
- **Function detection** - Proper function boundaries (objdump: symbol table only)
- **Control flow edges** - Jump/call/return relationships (objdump: none)
- **Type recovery** - Inferred data types (objdump: none)
- **Dead code detection** - Unreachable blocks (objdump: shows all bytes)

**When to use Ghidra:**
- Pattern differential tracking (need accurate basic block counts)
- CFG complexity metrics (loop depth, domination)
- Control flow optimization analysis (straightening, merging)

**When objdump is sufficient:**
- Simple instruction counting
- Opcode frequency analysis
- Quick sanity checks

## Related Work

- **Rainbow Table** (`tblgen-optimization-rainbow-table.md`) - Uses differential equation model for cost
- **Idempotent Pattern Optimization** (`idempotent-pattern-optimization.md`) - Storage vs recompute tradeoffs
- **Autoencoder Classification** (pending) - Trains on differential data
- **Stage2 Architecture** (`architecture.md`) - Disassembly and attestation
- **Subsumption Engine** (`ADR-001`) - Pattern matching infrastructure

---

## Conclusion

**Stochastic cache items** (patterns that SON clips to nothing at higher -O levels) require differential tracking to capture.

**Strategy:**

1. Baseline @ -Og preserves SON IR structure in assembly
2. Differential tracking measures cost reduction across -O levels
3. Stochastic patterns flagged by low survival rates
4. TableGen generation filters for stable patterns (survival ≥ 50%)
5. Differential data feeds autoencoder training

**Foundation required:** -Og baseline at regression-time (Phase 1).

**Late Stage2 goal:** Disasm → TableGen pattern extraction (Phase 3).

**Differential tracking enables:** Quantifying SON optimization effectiveness and building comprehensive pattern databases for ML-driven classification.
