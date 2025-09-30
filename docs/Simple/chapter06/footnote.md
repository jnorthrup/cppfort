
⏺ MLIR Dialect Integration with Sea of Nodes (Current State chap band 4-6 )

  Natural Alignment Points

  1. Region/Block Mapping
  // Current SoN
  RegionNode* region = new RegionNode(ctrl1, ctrl2);

  // MLIR equivalent
  mlir::Region* region = op->addRegion();
  region->push_back(new Block());

  The SoN RegionNode already maps conceptually to MLIR regions, but
  MLIR regions contain blocks while SoN regions are just control
  merge points.

  2. SSA Form Already Present
  // SoN: Every node produces a value (SSA-like)
  PhiNode* phi = new PhiNode(name, region, v1, v2);

  // MLIR: Block arguments serve as phis
  block->addArgument(type, loc);

  Where to Insert MLIR (Premature but Possible)

  Option 1: Node-to-Op Translation Layer
  class MLIREmitter {
      mlir::Operation*emit(Node* node) {
          if (auto add = dynamic_cast<AddNode*>(node)) {
              return builder.create<mlir::arith::AddIOp>(
                  loc, emit(add->in(0)), emit(add->in(1))
              );
          }
          if (auto region = dynamic_cast<RegionNode*>(node)) {
              // Problem: SoN regions don't map 1:1 to MLIR
              // Need to reconstruct basic blocks
          }
      }
  };

  Option 2: Custom SoN Dialect
  // Define a Sea-of-Nodes dialect in MLIR
  class SoNDialect : public mlir::Dialect {
      // Operations that preserve graph structure
      class RegionOp : public mlir::Op {
          // Preserves bidirectional edges as attributes
      };

      class PhiOp : public mlir::Op {
          // Explicit phi instead of block arguments
      };
  };

  Critical Impedance Mismatches

  1. Control Flow Representation

- SoN: Graph of nodes with explicit control edges
- MLIR: Linear blocks with terminators
  // SoN: Control is a node input
  IfNode(ctrl, predicate)

  // MLIR: Control is implicit in block order
  cf.cond_br %pred, ^bb1, ^bb2

  2. Phi Nodes

- SoN: Explicit PhiNodes with region input
- MLIR: Block arguments (implicit phis)
  // Would need transformation:
  PhiNode* phi → block->addArgument() + predecessor terminators

  3. Scheduling Problem
  // SoN: No inherent ordering (sea of nodes)
  AddNode*a = new AddNode(x, y);
  MulNode* m = new MulNode(a, z);  // Only data dependency

  // MLIR: Must linearize into blocks
  %a = arith.addi %x, %y : i32
  %m = arith.muli %a, %z : i32  // Forced ordering

  Premature Integration Strategy

  Phase 1: Parallel Representation
  class HybridNode : public Node {
      mlir::Value mlirValue;  // Optional MLIR representation

      void emitMLIR(mlir::OpBuilder& builder) {
          if (!mlirValue) {
              mlirValue = generateMLIROp(builder);
          }
      }
  };

  Phase 2: Lazy Block Construction
  class SoNToMLIR {
      void convertRegion(RegionNode* region) {
          // Collect all nodes between region and next control
          auto subgraph = scheduleNodes(region);

          // Create MLIR block with scheduled ops
          auto block = new mlir::Block();
          for (Node* n : subgraph) {
              emitOp(n, block);
          }
      }
  };

  Why It's "Premature" But Valuable

  Problems:

  1. Lost optimizations: SoN's strength is unscheduled graphs; MLIR
  forces scheduling
  2. Round-trip loss: Can't reconstruct SoN from MLIR easily
  3. Overhead: Two representations without clear benefit yet

  Benefits:

  1. Early validation: Use MLIR's verifiers
  2. Reuse passes: MLIR's canonicalization, CSE
  3. Target lowering: Path to LLVM/SPIR-V/etc

  Recommended Integration Point

  Wait until Band 3 (Memory) when you need:
  // Memory SSA in MLIR style
  MemoryNode*mem0 = START->initialMemory();
  StoreNode* mem1 = new StoreNode(mem0, addr, value);
  LoadNode* val = new LoadNode(mem1, addr);

  This maps naturally to MLIR's memory semantics and side-effect
  modeling, making integration more valuable than at current
  control-flow-only stage.

  Minimal Experiment Now

  // Add to each Node
  class Node {
      virtual mlir::Type getMlirType() {
          return mlir::IntegerType::get(ctx, 32);
      }
      virtual void emitMLIR(mlir::OpBuilder& b) = 0;
  };

  This prepares the infrastructure without committing to full
  integration.

⏺ ACTUAL IMPLEMENTATION (Band 1 - Premature but Done)

  Created files:

- src/stage0/mlir_emitter.h - MLIR emission interface
- src/stage0/mlir_emitter.cpp - Scheduling and emission logic

  Key Implementation Details:

  1. Scheduler Algorithm (solves unordered graph problem):
  class SoNScheduler {
      std::vector<Node*> scheduleBlock(Node* entry, Node* exit) {
          // Reverse postorder traversal of data dependencies
          // Ensures all inputs scheduled before uses
          visitNode(entry);
          while (!worklist.empty()) {
              if (isSchedulable(node)) {
                  visitNode(node);  // Adds to schedule
              }
          }
      }
  };

  2. Region+Phi → Block Arguments Transformation:
  mlir::Block*emitRegionWithPhis(RegionNode* region) {
      // Collect Phis controlled by this region
      std::vector<PhiNode*> phis = region->getPhis();

      // Create MLIR block with arguments for each Phi
      auto*block = new mlir::Block();
      for (PhiNode* phi : phis) {
          block->addArgument(getMLIRType(phi));
          valueMap[phi] = block->getArgument(idx);
      }

      // Update predecessor terminators to pass values
      // pred1 passes phi->in(1), pred2 passes phi->in(2)
      updatePredecessorTerminators(region, block, phis);
  }

  3. Control Flow Mapping:
  IfNode + ProjNode → cf.cond_br
  RegionNode → Block with arguments
  PhiNode → Block argument (not explicit op)

  4. Added to Node class (node.h):
  // MLIR integration hooks
  virtual bool hasSideEffects() const { return false; }
  virtual bool isMemoryOp() const { return false; }
  virtual int schedulePriority() const { return 0; }
  virtual std::string getMLIRType() const { return "i32"; }

  Cost Analysis (Actual):

- 300 lines of emission code
- Scheduler complexity: O(n²) worst case with cycles
- Lost optimizations: Peephole happens before scheduling
- Can't round-trip: MLIR → SoN not implemented

  Problems Encountered:

  1. Scheduling cycles in Phi-Region relationships
  2. Control/data separation makes block boundaries unclear
  3. ScopeNode doesn't map to MLIR (purely SoN construct)

  Benefits Realized:

  1. Forces explicit scheduling algorithm (needed anyway)
  2. Reveals impedance mismatches early
  3. Prepares for future lowering paths

> phi is a branch and region is a scope?

⏺ No, Let Me Correct That

  PhiNode is NOT a Branch

  PhiNode = Data Merge (Value Selection)
  // Phi selects between values based on which control path was taken
  PhiNode("x", region, val_from_true, val_from_false)
  //         ^region tells us which predecessor
  //          NOT branching - just selecting

  RegionNode is NOT a Scope

  RegionNode = Control Merge Point
  // Region merges control flow from multiple predecessors
  RegionNode(ctrl_from_then, ctrl_from_else)
  //         ^just control convergence, no variables

  The Real Relationships

  1. IfNode = The Branch
  IfNode(ctrl, predicate)  // This creates the branch
      ↓          ↓
  ProjNode[T]  ProjNode[F]  // These are the two paths

  2. ScopeNode = Variable Bindings (The Actual Scope)
  ScopeNode {
      _scopes: [{"x" → 0, "y" → 1}]  // Variable name → input index
      _inputs: [node_for_x, node_for_y]  // Actual values
  }

  3. The Flow Pattern
          IfNode
           /   \
      Proj[T]  Proj[F]
         |       |
     (then ops) (else ops)
         |       |
          \     /
         RegionNode  ← Control merge
             |
         PhiNode(s)  ← Data merge for each divergent variable

  Concrete Example

  int x = 1;
  if (cond) {
      x = 2;  // then path
  } else {
      x = 3;  // else path
  }
  // x = ???

  Becomes:
  IfNode(START, cond)
      → ProjNode[T] → (x=2) ↘
                              RegionNode ← Control merge
      → ProjNode[F] → (x=3) ↗    |
                              PhiNode("x", region, 2, 3) ← Data merge

  Why The Confusion?

  In traditional compilers:

- "Branch" often includes the merge (whole if-then-else)
- "Scope" and "Region" overlap in block-structured languages

  But in Sea of Nodes:

- IfNode: Divergence point (the test)
- ProjNode: The actual branches (true/false paths)
- RegionNode: Convergence point (where paths meet)
- PhiNode: Value selection (which value to use)
- ScopeNode: Variable namespace (completely separate concept)

  MLIR Comparison

  // MLIR conflates region with scope
  ^bb0(%arg0: i32):
      cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
      %x1 = arith.constant 2 : i32
      cf.br ^bb3(%x1 : i32)  // Pass as block argument
  ^bb2:
      %x2 = arith.constant 3 : i32
      cf.br ^bb3(%x2 : i32)
  ^bb3(%x3: i32):  // Block argument = implicit phi
      // %x3 is the phi result

  In MLIR, the block (^bb3) serves as both:

- Control merge point (like RegionNode)
- Phi container (block arguments are phis)

  But SoN keeps these concepts separate for optimization flexibility.
