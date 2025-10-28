# Cppfort Aspirations: A C++2 Transpiler Journey

## Project Overview

Cppfort is a pattern-driven transpiler that converts C++2 (cpp2) syntax - a cleaner, more expressive variant of C++ - into standard C++20 code. The project aims to achieve self-hosting through incremental, honest implementation without shortcuts or architecture astronautics.

```mermaid
mindmap
  root((Cppfort))
    Core Goal
      Self-Hosting Transpiler
      Pattern-Driven Transformation
      No Hacks or Cheating
    Input
      C++2 Syntax
      Function Declarations
      Parameter Modes
      Contracts
      Inspect Expressions
    Output
      Standard C++20
      Include Generation
      Forward Declarations
      Type Transformations
    Architecture
      YAML Pattern Loading
      Anchor-Based Matching
      Orbit System
      Bidirectional Patterns
```

## Current Status

As of October 2025, the project is in early stages with minimal working functionality.

```mermaid
pie title Test Status (Honest Assessment)
    "Passing Tests" : 0
    "Failing Tests" : 192
```

```mermaid
pie title Feature Completion
    "Working Features" : 13
    "Missing Features" : 87
```

### Reality Check Metrics
- **Working Features**: 3/8 (37.5%) - simple main, template alias, walrus operator
- **Passing Tests**: 0/192 (0%)
- **Can Transpile Hello World**: NO
- **Can Self-Host**: NO
- **Lines of Working Code**: ~500 of 5000+

## Implementation Roadmap

The project follows a strict "one feature at a time" approach with progress gates.

```mermaid
timeline
    title Implementation Stages
    section Stage 1 (Week 1)
      Make ANYTHING Work
      : Pass ONE test completely
      : Simplest function transformation
      : No hardcoding, no regex hacks
    section Stage 2 (Week 2)
      Variable Declarations
      : Typed variables (x: int = 42)
      : Auto variables (x := 42)
      : Pattern-driven transformation
    section Stage 3 (Week 3)
      Parameters
      : Basic parameters
      : Parameter modes (in/out/inout/move/forward)
      : Multiple parameters
    section Stage 4 (Week 4)
      Include Generation
      : Detect std:: usage
      : Generate #include directives
      : Avoid duplicates
```

## Core Aspirations

### 1. Honest Implementation
No fake progress, no marking incomplete features as done, no architecture without working code.

```mermaid
flowchart TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Test passes?]
    B -->|No| D[Fix it]
    C -->|Yes| E[Move forward]
    C -->|No| F[Document failure]
    F --> D
    D --> B
```

### 2. Pattern-Driven Transformation
All transformations must be driven by declarative patterns, not hardcoded logic.

```mermaid
flowchart LR
    A[YAML Patterns] --> B[Pattern Loader]
    B --> C[Anchor Extraction]
    C --> D[Segment Matching]
    D --> E[Substitution Templates]
    E --> F[C++ Output]
```

### 3. Self-Hosting Path
Ultimate goal: the transpiler can compile itself.

```mermaid
flowchart TD
    A[Stage 0: Basic Transpiler] --> B[Transpile Stage 1]
    B --> C[Stage 1: Enhanced Transpiler]
    C --> D[Transpile Stage 2]
    D --> E[Stage 2: More Features]
    E --> F[Transpile Full Cppfort]
    F --> G[Self-Hosting Achieved]
```

## Feature Aspirations

### Level 0: Minimal Viable (REQUIRED)
```mermaid
mindmap
  root((Level 0))
    Function Declaration Parsing
      Name, parameters, return type
    Function Body Processing
      Variable declarations
    Include Preservation
      Keep existing includes
```

### Level 1: Basic Parameters
```mermaid
mindmap
  root((Level 1))
    Inout Parameters
      inout s: string → string& s
    Function Ordering
      Forward declarations
    Parameter Modes
      in, out, move, forward
```

### Level 2: Type System
```mermaid
mindmap
  root((Level 2))
    Auto Deduction
      x := 42 → auto x = 42
    Template Syntax
      Type aliases
    Contract Syntax
      Pre/post conditions
```

### Level 3: Advanced Features
```mermaid
mindmap
  root((Level 3))
    Inspect Expressions
      Pattern matching
    UFCS
      Uniform function call syntax
    Contracts
      Design by contract
```

## Implementation Gaps

Critical missing functionality that prevents basic transpilation.

```mermaid
flowchart TD
    A[Input CPP2] --> B{Parameter Transformation?}
    B -->|NO| C[FAIL: Cannot handle parameters]
    B -->|YES| D{Include Generation?}
    D -->|NO| E[FAIL: Missing headers]
    D -->|YES| F{Forward Declarations?}
    F -->|NO| G[FAIL: Function ordering]
    F -->|YES| H{Bidirectional Patterns?}
    H -->|NO| I[FAIL: Round-trip conversion]
    H -->|YES| J{Recursive Processing?}
    J -->|NO| K[FAIL: Nested patterns]
    J -->|YES| L[SUCCESS: Full Transpilation]
```

## Success Metrics

### Milestones
```mermaid
gantt
    title Project Milestones
    dateFormat  YYYY-MM-DD
    section Minimal
        Milestone 1: 1 test passing    :done, m1, 2025-10-01, 1d
        20% features working          :done, f1, 2025-10-01, 1d
        Hello world transpilation      :done, h1, 2025-10-01, 1d
    section Basic
        Milestone 2: 10 tests passing  :active, m2, 2025-11-01, 30d
        40% features working          :f2, 2025-11-01, 30d
        Simple programs                :s1, 2025-11-01, 30d
    section Functional
        Milestone 3: 50 tests passing  :m3, 2025-12-01, 60d
        60% features working          :f3, 2025-12-01, 60d
        Most programs                  :s2, 2025-12-01, 60d
    section Complete
        Milestone 4: 150+ tests passing :m4, 2026-02-01, 90d
        90%+ features working         :f4, 2026-02-01, 90d
        Self-hosting                   :sh, 2026-02-01, 90d
```

## Anti-Patterns to Avoid

### What NOT to Do
```mermaid
flowchart TD
    A[Common Pitfalls] --> B[Architecture Astronautics]
    A --> C[Fake Progress]
    A --> D[Scope Creep]
    A --> E[Regex Hacks]
    
    B --> B1[Semantic codec design]
    B --> B2[N-way graph mapping]
    B --> B3[Orbit recursion theory]
    
    C --> C1[Mark incomplete as done]
    C --> C2[Claim working when crashing]
    C --> C3[100% confidence when 0% works]
    
    D --> D1[Bidirectional before basic]
    D --> D2[Self-hosting before working]
    D --> D3[Performance before correctness]
    
    E --> E1[Post-processing hacks]
    E --> E2[Hardcoded solutions]
    E --> E3[Regex instead of patterns]
```

## The Truth Test

The ultimate measure of progress: can it transpile and run real code?

```mermaid
flowchart TD
    A[CPP2 Input] --> B[Transpile to C++]
    B --> C[Compile with g++ -std=c++20]
    C --> D[Run executable]
    D --> E{Exit code 0?}
    E -->|YES| F[SUCCESS]
    E -->|NO| G[FAILURE - Keep working]
```

## Daily Progress Ritual

```
Date: YYYY-MM-DD
Tests Passing: X/192
Features Complete: [Actual working features]
Current Stage: X.X
Blocking Issue: [What's preventing next test]
Lines Changed: +X -Y
Honest Assessment: [Can it transpile anything useful?]
```

## Core Mantra

**Make it work, make it right, make it fast - IN THAT ORDER**

Currently: Nothing works.  
Goal: Make something work.  
Everything else can wait.