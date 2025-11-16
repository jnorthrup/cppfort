## Nov 2025 — Graph-first architecture implementation updates

- Added `.clang-format` to revive coding standards
- Implemented `cpp2_cas` placeholder with tests to canonicalize cpp2 markdown blocks
- Introduced `GraphMatcher` stub and `JsonScanner` standalone test
- Created `scripts/audit_regex_usage.sh` and audited regex usage for planful migration

# Progress Summary: Orbit-Based Transpiler Pipeline

## ✅ Completed Tasks

### 1. JSON ↔ YAML Two-Way Converter
- **Tool**: `/tools/json_yaml` (wrapper) and `/tools/two_way_json_yaml.py` (core)
- **Functionality**: 
  - Converts JSON ↔ YAML preserving all semantic structure
  - Supports round-trip conversion (JSON → YAML → JSON is lossless)
  - Handles pattern files with templates, anchors, and evidence types
- **Usage**:
  ```bash
  # Convert JSON to YAML for editing
  tools/json_yaml --json-to-yaml patterns/cppfort_core_patterns.json patterns/cppfort_core_patterns.yaml
  
  # Convert YAML back to JSON for machine processing
  tools/json_yaml --yaml-to-json patterns/cppfort_core_patterns.yaml patterns/cppfort_core_patterns.json
  
  # Create both formats automatically
  tools/json_yaml --backup cppfort_core_patterns
  ```

### 2. Enhanced WideScanner with TypeEvidence (Step 1)
- **Location**: `/src/stage0/wide_scanner.h`, `/src/stage0/wide_scanner.cpp`
- **Status**: Enhanced existing `wide_scanner` with:
  - SIMD-accelerated boundary detection (ARM NEON + x86 AVX/SSE)
  - TypeEvidence integration at boundary level
  - Orbit metadata support (lattice_mask + confidence)
- **Key Features**:
  - UTF-8 boundary awareness
  - Delimiter detection (semicolons, commas, braces, brackets, parens)
  - XAI 4.2 orbit support with 5-anchor tuple detection

### 3. JSON Orbit Scanner & YAML Orbit Scanner (New)
- **Files Created**:
  - `/src/stage0/json_orbit_scanner.h` - Orbit-based JSON pattern detection
  - `/src/stage0/yaml_orbit_scanner.h` - Orbit-based YAML pattern detection
  - Uses same base as `OrbitScanner` but language-specific patterns

### 4. RegionNode Graph Structure (Step 2 - Terraced Field)
- **File**: `/src/stage0/region_node.h`
- **Architecture**: Direct MLIR mapping with:
  - **RegionNode**: Maps to `mlir::Region` or `mlir::Block`
  - **Operation**: Light-weight MLIR operation stub
  - **Value**: SSA value representation
- **Region Types**: FUNCTION, BLOCK, NAMED_REGION, CONDITIONAL, LOOP, INITIALIZER, RETURN_REGION
- **Features**:
  - Hierarchical terraced structure
  - SSA form preparation
  - MLIR dialect metadata
  - Source location tracking
  - Orbit confidence scores

## 🔧 In Progress

### 5. RBCursiveScanner → Structural Inference Engine (Step 3)
- **Goal**: Repurpose RBCursiveScanner for region carving
- **Key Method**: `carve_regions(const std::vector<BoundaryEvent>& events)`
- **Algorithm**: "Wobbling Window" for confix deduction
  - Start with high-confidence anchors (e.g., `{`)
  - Widen/slide scanning forward with confix depth tracking
  - Contract/perturb to find valid boundaries
  - Recursive terracing for nested structures

## ⏳ Pending Tasks

### 6. PatternApplier (Step 4)
- **Purpose**: Semantic labeling (not text substitution)
- **Approach**: Match patterns for classification
- **Action**: Populate RegionNode structure with semantic types
- **Example**: "cpp2_function_definition" → `func.func` with parameter extraction

### 7. GraphToMlirWalker (Step 5)
- **Purpose**: Walk RegionNode graph to generate MLIR
- **Integration**: Use `cpp2_mlir_assembler.cpp` with `mlir::OpBuilder`
- **Mapping**: Direct 1-to-1 from graph to MLIR constructs

### 8. Cleanup & Test Updates (Step 6)
- **Remove**: `cpp2_emitter.cpp`, `depth_pattern_matcher.cpp`
- **Update**: `test_reality_check.cpp` with new pipeline
- **Pipeline**: WideScanner → RBCursiveScanner → PatternApplier → GraphToMlirWalker

### 9. Character-Class Inference
- **Focus**: TypeEvidence span analysis
- **Goal**: Infer syntactic categories from character class patterns
- **Next**: Implement character-class inference algorithms

## 📁 Key Files & Architecture

### Pattern Files
```
patterns/
├── cppfort_core_patterns.json    # Core Cpp2 patterns (machine-readable)
├── cppfort_core_patterns.yaml    # Human-editable version
├── json_grammar.json            # JSON structure patterns
├── json_grammar.yaml            # YAML version
├── semantic_units.json          # Semantic unit definitions
└── bnfc_cpp2_complete.yaml      # Legacy BNFC patterns
```

### Scanner Architecture
```
src/stage0/
├── wide_scanner.h/cp            # SIMD-accelerated boundary detection
├── orbit_scanner.h/cpp          # Generic orbit-based pattern matching
├── json_orbit_scanner.h         # JSON-specific orbit scanner (NEW)
├── yaml_orbit_scanner.h         # YAML-specific orbit scanner (NEW)
├── json_scanner.h               # JSON structure parser
├── yaml_scanner.h               # YAML structure parser (NEW)
└── region_node.h                # Terraced field MLIR mapping (NEW)
```

### Core Evidence & Analysis
```
src/stage0/
├── evidence.h                   # EvidenceSpan with confidence metrics
├── type_evidence.h              # Hierarchical character class counters
├── lattice_classes.h            # Byte-level character classification
└── confix_orbit.h/cpp           # Confix tracking with mementos
```

## 🚀 Next Steps

1. **Implement RBCursiveScanner structural carving** (`carve_regions` method)
2. **Extend WideScanner TypeEvidence integration** for boundary events
3. **Create PatternApplier** for semantic classification (not text rewriting)
4. **Implement GraphToMlirWalker** to traverse RegionNode tree
5. **Test end-to-end pipeline** with simple Cpp2 functions
6. **Add character-class inference** on EvidenceSpan ranges

## 📊 Current State

- **12 pattern files** in JSON/YAML formats (6 core + 6 converted)
- **5 active scanners** (wide, orbit, json_orbit, yaml_orbit, base pattern scanner)
- **1 terraced field architecture** implemented (RegionNode)
- **2-way converter** operational (JSON ↔ YAML)
- **Evidence system** fully integrated
