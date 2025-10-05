# Stage0 Orbit Scanner Architecture

## Overview

The stage0 orbit scanner implements a multi-layer architecture for grammar detection and transpilation using SIMD-accelerated scanning, orbit-based structural analysis, and evidence-based pattern matching.

## Architecture Diagram

```mermaid
classDiagram
    %% Core Scanner Classes
    class WideScanner {
        -OrbitContext orbit_context_
        +generateAlternatingAnchors(source) AnchorPoint[]
        +scanAnchorsSIMD(source, anchors) Boundary[]
        +scanAnchorsWithOrbits(source, anchors) Boundary[]
        +generateOrbitTuples(source, chunk_size) AnchorTuple[]
        +findBoundarySIMD(data, pos, remaining) size_t
        +isUTF8Boundary(data, pos) bool
        -hasDelimiter(data, len, delim) bool
        -findDelimiterMask(data, len) int
    }

    class OrbitScanner {
        -OrbitScannerConfig m_config
        -RabinKarp m_rabinKarp
        -OrbitContext m_context
        -IMultiGrammarLoader m_loader
        -CPP2KeyResolver m_cpp2Resolver
        +initialize() bool
        +scan(code) DetectionResult
        +scan(code, patterns) DetectionResult
        +getConfig() OrbitScannerConfig
        +updateConfig(config) void
        +getPatternCount() size_t
        +getLoadedGrammars() GrammarType[]
        -validateConfig() bool
        -validateInitialization() bool
        -findMatches(code, patterns) MatchResults
        -applyCPP2KeyResolution(code, candidates) MatchResults
        -detectOrbitPattern(grammar, orbitCounts, pos, code) double
        -analyzeMatches(matches) DetectionResult
        -calculateGrammarConfidence(grammar, matches) double
        -determineBestGrammar(scores) GrammarType
        -determine_scope_type(context) string
        -determine_lattice_mask(context) uint16_t
        -generateReasoning(result) string
    }

    class OrbitContext {
        -int _braceDepth
        -int _bracketDepth
        -int _angleDepth
        -int _parenDepth
        -int _quoteDepth
        -int _numberDepth
        -bool _inNumber
        -size_t _maxDepth
        -OrbitMatch[] _matches
        +processMatch(match) bool
        +isBalanced() bool
        +depth(type) int
        +getDepth() int
        +getMaxDepth() size_t
        +calculateConfidence() double
        +update(ch) void
        +getCounts() array~size_t_6~
        +confixMask() uint8_t
        +matches() OrbitMatch[]
        +reset() void
        +wouldBeValid(match) bool
    }

    %% Data Structures
    class AnchorPoint {
        +size_t position
        +size_t spacing
        +bool is_utf8_boundary
    }

    class Boundary {
        +size_t position
        +char delimiter
        +bool is_delimiter
        +uint16_t lattice_mask
        +double orbit_confidence
    }

    class AnchorTuple {
        +EvidenceAnchor[5] anchors
        +span~char~ evidence_range
        +double composite_confidence
        +interleave_evidence() void
        -count_delimiters(span) int
        -detect_indentation(span) int
        -detect_numeric_patterns(span) bool
        -detect_legal_classes(span) bool
        -detect_cascading_ranges(span) bool
    }

    class EvidenceAnchor {
        +AnchorType anchor_type
        +xai_variant value
        +double confidence
        +span~char~ span_range
        +size_t position
        +operator+(other) EvidenceAnchor
        +operator|(other) EvidenceAnchor
        +operator&(other) EvidenceAnchor
    }

    class OrbitMatch {
        +string patternName
        +GrammarType grammarType
        +size_t startPos
        +size_t endPos
        +double confidence
        +string signature
        +uint64_t[6] orbitHashes
        +size_t[6] orbitCounts
    }

    class OrbitPattern {
        +string name
        +uint32_t orbit_id
        +string[] signature_patterns
        +string[] protocol_indicators
        +string[] version_patterns
        +double weight
        +int expected_depth
        +string required_confix
        +uint8_t confix_mask
        +uint8_t grammar_modes
        +uint16_t lattice_filter
        +string[] prev_tokens
        +string[] next_tokens
        +string scope_requirement
    }

    class DetectionResult {
        +GrammarType detectedGrammar
        +double confidence
        +string reasoning
        +map~GrammarType_double~ grammarScores
        +MatchResults matches
    }

    class OrbitScannerConfig {
        +filesystem::path patternsDir
        +double patternThreshold
        +size_t maxMatches
        +size_t maxDepth
    }

    %% Enumerations
    class AnchorType {
        <<enumeration>>
        COUNT_DELIMITERS
        INDENTATION
        NUMBER_DUCK_TYPE
        LEGAL_CLASSES
        CASCADING_RANGES
    }

    class OrbitType {
        <<enumeration>>
        None
        OpenBrace
        CloseBrace
        OpenBracket
        CloseBracket
        OpenAngle
        CloseAngle
        OpenParen
        CloseParen
        Quote
        NumberStart
        NumberEnd
        Unknown
    }

    class GrammarType {
        <<enumeration>>
        C
        CPP
        CPP2
        UNKNOWN
    }

    %% Relationships
    WideScanner --> OrbitContext : uses
    WideScanner --> AnchorPoint : generates
    WideScanner --> Boundary : produces
    WideScanner --> AnchorTuple : generates

    OrbitScanner --> OrbitContext : owns
    OrbitScanner --> OrbitScannerConfig : configured by
    OrbitScanner --> DetectionResult : returns
    OrbitScanner --> OrbitPattern : uses

    OrbitContext --> OrbitMatch : stores
    OrbitContext --> OrbitType : tracks

    AnchorTuple --> EvidenceAnchor : contains 5
    EvidenceAnchor --> AnchorType : typed by

    OrbitMatch --> GrammarType : categorizes
    OrbitPattern --> GrammarType : targets
    DetectionResult --> GrammarType : identifies

    Boundary --> OrbitContext : feeds into

    note for WideScanner "SIMD-accelerated scanner\nUTF-8 boundary detection\nAlternating anchor generation"
    note for OrbitScanner "Multi-grammar detection\nPattern matching\nCPP2 key resolution"
    note for OrbitContext "Structural balance tracking\nDepth management\nConfix masking"
    note for AnchorTuple "5-anchor evidence system\nComposite confidence\nXAI 4.2 integration"
```

## Event Flow: Scanning and Transpilation

```mermaid
sequenceDiagram
    participant Main as main.cpp
    participant WS as WideScanner
    participant OC as OrbitContext
    participant AT as AnchorTuple
    participant Boundary as Boundary[]

    Note over Main: PHASE 1: Anchor Generation
    Main->>WS: generateAlternatingAnchors(source)
    activate WS
    WS->>WS: Detect UTF-8 boundaries
    WS->>WS: Alternate 64/32 byte spacing
    WS-->>Main: AnchorPoint[]
    deactivate WS

    Note over Main: PHASE 2: Orbit Scanning
    Main->>WS: scanAnchorsWithOrbits(source, anchors)
    activate WS
    WS->>WS: SIMD scan between anchors
    loop For each delimiter found
        WS->>OC: update(delimiter)
        activate OC
        OC->>OC: Update depth counters
        Note over OC: EVENT: FIRE (open delim)<br/>or RING (close delim)
        OC->>OC: Calculate confix mask
        OC-->>WS: Context updated
        deactivate OC
    end
    WS->>WS: Generate boundaries with orbit data
    WS-->>Main: Boundary[]
    deactivate WS

    Note over Main: PHASE 3: Ring Building
    Main->>Main: Build rings from boundaries
    loop For each boundary
        alt Open delimiter: { ( [ <
            Main->>Main: Push to stack
            Note over Main: EVENT: FIRE<br/>depth++
        else Close delimiter: } ) ] >
            Main->>Main: Pop from stack
            Main->>Main: Create Ring{open, close}
            Note over Main: EVENT: RING<br/>depth--
        end
    end

    Note over Main: PHASE 4: Tree Construction
    Main->>Main: Build TreeNode from rings
    Main->>Main: Link parent/child relationships
    Main->>Main: Propagate context bitmasks
    loop For each position
        Main->>Main: pos_mask[p] |= node.mask
        Main->>Main: pos_confidence[p] = max(conf)
    end

    Note over Main: PHASE 5: Grammar Detection
    Main->>Main: Count ring types
    Main->>Main: Classify grammar by orbit structure
    alt brace_rings > 0 && angle_rings < brace_rings/4
        Main->>Main: detected_grammar = C
    else angle_rings > brace_rings/4
        Main->>Main: detected_grammar = CPP
    else cpp2_markers > boundaries/10
        Main->>Main: detected_grammar = CPP2
    end

    Note over Main: PHASE 6: Output
    Main->>Main: Write transpiled output
    Main-->>Main: Return grammar metadata
```

## Advanced Pattern Matching Flow

```mermaid
sequenceDiagram
    participant Client as Client Code
    participant OS as OrbitScanner
    participant OC as OrbitContext
    participant RK as RabinKarp
    participant CPP2 as CPP2KeyResolver
    participant DB as PatternDatabase

    Note over Client: Initialize Scanner
    Client->>OS: new OrbitScanner(config, loader)
    OS->>OC: new OrbitContext(maxDepth)
    OS->>RK: Initialize RabinKarp
    OS->>CPP2: Initialize CPP2KeyResolver
    Client->>OS: initialize()
    OS->>DB: Load patterns from directory
    DB-->>OS: Patterns loaded

    Note over Client: Scan Code
    Client->>OS: scan(code)
    activate OS

    OS->>RK: findMatches(code, patterns)
    activate RK
    loop For each pattern
        RK->>RK: Calculate hash signature
        RK->>OC: Check confix mask
        OC-->>RK: Context valid
        RK->>RK: Check lattice filter
        RK->>RK: Check grammar mode
        alt Match found
            RK->>RK: Create OrbitMatch
        end
    end
    RK-->>OS: MatchResults (candidates)
    deactivate RK

    OS->>CPP2: applyCPP2KeyResolution(code, candidates)
    activate CPP2
    loop For each candidate
        CPP2->>CPP2: Determine scope type
        CPP2->>CPP2: Check lattice mask
        CPP2->>CPP2: Apply context windows
        alt Context valid
            CPP2->>CPP2: Enhance confidence
        else Context invalid
            CPP2->>CPP2: Filter out match
        end
    end
    CPP2-->>OS: Refined MatchResults
    deactivate CPP2

    OS->>OS: analyzeMatches(matches)
    activate OS
    loop For each GrammarType
        OS->>OS: calculateGrammarConfidence(grammar, matches)
        OS->>OS: Count pattern matches
        OS->>OS: Weight by pattern importance
        OS->>OS: Factor orbit structure
    end
    OS->>OS: determineBestGrammar(scores)
    OS->>OS: generateReasoning(result)
    OS-->>OS: DetectionResult
    deactivate OS

    OS-->>Client: DetectionResult
    deactivate OS
```

## Event Types and Their Significance

### FIRE Events
**Triggered by:** Opening delimiters `{`, `(`, `[`, `<`, `"`

**Actions:**
- Push delimiter position to corresponding stack
- Increment depth counter for delimiter type
- Update confix context mask
- Record position in orbit tracking

**Significance:** Marks the beginning of a structural orbit region

### RING Events
**Triggered by:** Closing delimiters `}`, `)`, `]`, `>`, `"`

**Actions:**
- Pop matching opening delimiter from stack
- Create Ring structure with {open_pos, close_pos, delim, depth}
- Decrement depth counter
- Update confix context mask
- Calculate orbit confidence based on depth

**Significance:** Completes a structural orbit, enabling grammar classification

### Boundary Detection
**Triggered by:** SIMD scan finding delimiters or UTF-8 boundaries

**Data captured:**
- Position in source
- Delimiter character
- Lattice mask (byte-level classification)
- Orbit confidence from context

**Significance:** Provides anchor points for ring construction and evidence gathering

### Grammar Classification
**Triggered by:** Analysis of ring patterns after full scan

**Decision criteria:**
- C: High brace/paren ratio, low angle brackets
- CPP: Significant angle brackets (templates)
- CPP2: High density of `:` and `=` markers

**Significance:** Determines transpilation strategy and output format

### Context Switching via Bitmasks
**Triggered by:** FIRE and RING events

**Bitmask structure (confixMask):**
```
Bit 0: TopLevel (depth == 0)
Bit 1: InBrace  ({...})
Bit 2: InParen  ((...))
Bit 3: InAngle  (<...>)
Bit 4: InBracket ([...])
Bit 5: InQuote   ("...")
```

**Significance:** Enables context-aware pattern matching and disambiguation

## XAI 4.2 Anchor System

### Five Anchor Types

1. **COUNT_DELIMITERS**: Array bounds, loop constructs, repetition patterns
2. **INDENTATION**: Scope-based evidence from whitespace patterns
3. **NUMBER_DUCK_TYPE**: Numeric literal classification and type inference
4. **LEGAL_CLASSES**: Valid C++2/CPP2 type constructs and metaclass patterns
5. **CASCADING_RANGES**: Hierarchical evidence propagation through spans

### Evidence Interleaving

All 5 anchor types fire concurrently on the same span, each contributing:
- Individual confidence score (0.0-1.0)
- Evidence value (count, span, or score)
- Position and span range

**Composite confidence** = Average of all 5 anchor confidences

### Evidence Operators

- `anchor1 + anchor2`: Merge spans, average confidence
- `anchor1 | anchor2`: Alternative, take higher confidence
- `anchor1 & anchor2`: Intersection, minimum confidence

## Key Architectural Patterns

### Alternating Anchor Strategy
Generates anchor points at UTF-8 boundaries with alternating 64/32 byte spacing, optimizing for SIMD processing while maintaining character boundary alignment.

### Hierarchical Orbit Tracking
Maintains separate depth counters for each delimiter type, enabling precise structural analysis and pattern disambiguation.

### N-Way Grammar Detection
Concurrent evaluation of multiple grammar hypotheses (C, CPP, CPP2) with confidence scoring and pattern-based disambiguation.

### Context-Aware Pattern Matching
Patterns specify required context via confix masks, depth constraints, and scope requirements, enabling accurate detection in ambiguous scenarios.

### Evidence-Based Confidence
Combines multiple evidence sources (orbit structure, pattern matches, context validity) into composite confidence scores for robust grammar detection.

## Performance Characteristics

- **SIMD Acceleration**: 16-byte parallel processing for boundary detection
- **UTF-8 Boundary Detection**: Efficient multi-byte character handling
- **Alternating Anchors**: Balances granularity with processing overhead
- **Orbit Caching**: Position-indexed masks and confidence scores
- **Pattern Pre-filtering**: Lattice masks reduce pattern search space

## File Locations

- Main transpiler: `/Users/jim/work/cppfort/src/stage0/main.cpp`
- SIMD scanner interface: `/Users/jim/work/cppfort/src/stage0/wide_scanner.h`
- Orbit scanner: `/Users/jim/work/cppfort/src/stage0/orbit_scanner.h`
- Orbit context and masks: `/Users/jim/work/cppfort/src/stage0/orbit_mask.h`
- XAI anchor types: `/Users/jim/work/cppfort/src/stage0/xai_orbit_types.h`
- Pattern definitions: `/Users/jim/work/cppfort/src/stage0/tblgen_patterns.h`
