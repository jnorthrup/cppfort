# Sea of Nodes Documentation Index

This directory contains the complete Sea of Nodes tutorial and documentation from Cliff Click's Simple compiler project.

## Overview

The Sea of Nodes is an intermediate representation (IR) pioneered by Cliff Click around 1990. It forms the core of industrial-strength compilers including:
- HotSpot's C2 compiler (Java)
- Google's V8 compiler (JavaScript)
- Sun/Oracle's Graal compiler

## Quick Links

- [Main README](README.md) - Project overview and chapter list
- [Type Analysis](TypeAnalysis.md) - Detailed type system documentation
- [A Simple Reply](ASimpleReply.md) - FAQ and common questions

## Tutorial Chapters

Each chapter builds a simple language compiler, progressively adding features while demonstrating Sea of Nodes concepts.

### Basics (Chapters 1-9)

| Chapter | Title | Description |
|---------|-------|-------------|
| [1](chapter01/README.md) | Introduction | Basic structure, nodes, graph representation |
| [2](chapter02/README.md) | Binary Arithmetic | Simple arithmetic with constants, constant folding |
| [3](chapter03/README.md) | Local Variables | Assignment statements, SSA form |
| [4](chapter04/README.md) | External Variables | Input variables, comparison operators |
| [5](chapter05/README.md) | If Statement | Control flow graphs, CFG construction |
| [6](chapter06/README.md) | Dead Code Elimination | Peephole optimization for dead control flow |
| [7](chapter07/README.md) | While Loops | Loop constructs, eager phi approach |
| [8](chapter08/README.md) | Loop Optimization | Lazy phi creation, break/continue |
| [9](chapter09/README.md) | Global Value Numbering | Iterative peepholes, worklists, GVN |

### Advanced Features (Chapters 10-18)

| Chapter | Title | Description |
|---------|-------|-------------|
| [10](chapter10/README.md) | User-defined Types | Struct types, memory edges, alias analysis |
| [11](chapter11/README.md) | Global Code Motion | Scheduling optimizations |
| [12](chapter12/README.md) | Float Type | Floating-point numbers and operations |
| [13](chapter13/README.md) | Nested References | Struct nesting, complex types |
| [14](chapter14/README.md) | Narrow Types | Bytes and other primitive types |
| [15](chapter15/README.md) | Arrays | Static arrays, load/store operations |
| [16](chapter16/README.md) | Constructors | Object construction syntax |
| [17](chapter17/README.md) | Syntax Sugar | var/val, compound assignment, for loops |
| [18](chapter18/README.md) | Functions and Calls | Function definitions, call/return |

### Code Generation (Chapters 19-24)

| Chapter | Title | Description |
|---------|-------|-------------|
| [19](chapter19/README.md) | Instruction Selection | Portable compilation, backend basics |
| [20](chapter20/README.md) | Register Allocation | Graph coloring register allocation |
| [21](chapter21/README.md) | Instruction Encoding | Machine code generation basics |
| [22](chapter22/README.md) | Hello World | Complete program compilation |
| [23](chapter23/README.md) | Methods and Types | Object-oriented features |
| [24](chapter24/README.md) | [Final Chapter] | Advanced topics and conclusions |

## Visualizations

The [docs](docs/) directory contains GraphViz diagrams (.gv files) and rendered SVG files that visualize the Sea of Nodes IR:
- Lattice diagrams for type analysis
- Control flow graphs
- Node dependency graphs

## Key Concepts

### Nodes
The Sea of Nodes represents a program as a graph of nodes:
- **Control Nodes**: Represent control flow (if, while, return, etc.)
- **Data Nodes**: Represent computations and values

### Edges
- **Def-Use Edges**: Connect definitions to their uses
- **Control Edges**: Represent control flow dependencies

### Benefits
- **Peephole Optimizations**: Can be performed during parsing
- **Global Optimizations**: Dependencies are explicit in the graph
- **Efficient Analysis**: Graph structure enables fast algorithms

## Further Reading

### Academic Papers
1. [From Quads to Graphs: An Intermediate Representation's Journey](http://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR93366-S.pdf)
2. [Combining Analyses, Combining Optimizations](https://dl.acm.org/doi/pdf/10.1145/201059.201061)
3. [A Simple Graph-Based Intermediate Representation](https://www.oracle.com/technetwork/java/javase/tech/c2-ir95-150110.pdf)
4. [Global Code Motion Global Value Numbering](https://courses.cs.washington.edu/courses/cse501/06wi/reading/click-pldi95.pdf)

### Presentations
- [Cliff Click — The Sea of Nodes and the HotSpot JIT](https://www.youtube.com/watch?v=9epgZ-e6DUU)
- [How a Sea of Nodes Compiler Works | Ben Titzer](https://www.youtube.com/watch?v=7hENC8mp77c)

## Implementation Notes

This tutorial is implemented in Java, but the concepts are language-agnostic. Each chapter contains:
- Complete source code
- Detailed explanations
- Visual diagrams
- Test cases

The implementation demonstrates how to build a compiler that uses the Sea of Nodes IR from the ground up, making it an excellent learning resource for understanding this powerful intermediate representation.