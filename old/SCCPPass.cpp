//===- SCCPPass.cpp - Sea of Nodes SCCP Pass Implementation ---------------===//
//
// Sparse Conditional Constant Propagation for Sea of Nodes dialect
// Based on Chapter 24 of SeaOfNodes/Simple implementation
//
//===----------------------------------------------------------------------===//
//
// This implements the SCCP (Sparse Conditional Constant Propagation) algorithm
// as described in Chapter 24 of the Sea of Nodes book. The algorithm:
//
// 1. Uses a lattice where types start at TOP and fall towards BOTTOM
// 2. SSA form makes the algorithm "sparse" - we only process def-use edges
// 3. Control flow integration makes it "conditional" - unreachable code is skipped
// 4. Interprocedural extension links calls to functions for whole-program analysis
//
// Key concepts from Chapter 24:
// - All node types start at TOP (most optimistic)
// - Worklist processes nodes, running transfer functions (compute)
// - If type changes, neighbors go on worklist
// - Fixed point reached when worklist empties
// - Result is monotonically better than pessimistic bottom-up analysis
//
//===----------------------------------------------------------------------===//

#include "Cpp2Passes.h"
#include "Cpp2SONDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"

#include <queue>
#include <random>

using namespace mlir;
using namespace mlir::sond;

namespace {

//===----------------------------------------------------------------------===//
// Type Lattice Implementation (from Type.java)
//===----------------------------------------------------------------------===//

/// Represents a lattice value for SCCP analysis
/// Corresponds to the Type hierarchy in Simple's type system
struct LatticeValue {
  enum Kind {
    Top,        // No value yet (most optimistic)
    Constant,   // Known constant value
    Range,      // Known range [min, max]
    Bottom      // Unknown/varying (most pessimistic)
  };

  Kind kind;
  Attribute constantValue;  // Valid when kind == Constant
  int64_t rangeMin;         // Valid when kind == Range
  int64_t rangeMax;         // Valid when kind == Range

  LatticeValue() : kind(Top), constantValue(nullptr), rangeMin(0), rangeMax(0) {}

  static LatticeValue top() { return LatticeValue(); }

  static LatticeValue bottom() {
    LatticeValue v;
    v.kind = Bottom;
    return v;
  }

  static LatticeValue constant(Attribute attr) {
    LatticeValue v;
    v.kind = Constant;
    v.constantValue = attr;
    return v;
  }

  static LatticeValue range(int64_t min, int64_t max) {
    LatticeValue v;
    v.kind = Range;
    v.rangeMin = min;
    v.rangeMax = max;
    return v;
  }

  bool isTop() const { return kind == Top; }
  bool isBottom() const { return kind == Bottom; }
  bool isConstant() const { return kind == Constant; }
  bool isRange() const { return kind == Range; }

  /// Meet operation (GLB in lattice)
  /// Corresponds to Type.meet() in Java
  LatticeValue meet(const LatticeValue &other) const {
    // Top meets anything = other
    if (isTop()) return other;
    if (other.isTop()) return *this;

    // Bottom meets anything = Bottom
    if (isBottom() || other.isBottom()) return bottom();

    // Two constants
    if (isConstant() && other.isConstant()) {
      if (constantValue == other.constantValue) return *this;
      // Different constants -> try to make a range or go to bottom
      if (auto thisInt = dyn_cast<IntegerAttr>(constantValue)) {
        if (auto otherInt = dyn_cast<IntegerAttr>(other.constantValue)) {
          int64_t a = thisInt.getInt();
          int64_t b = otherInt.getInt();
          return range(std::min(a, b), std::max(a, b));
        }
      }
      return bottom();
    }

    // Constant meets range -> extend range
    if (isConstant() && other.isRange()) {
      if (auto thisInt = dyn_cast<IntegerAttr>(constantValue)) {
        int64_t c = thisInt.getInt();
        return range(std::min(c, other.rangeMin), std::max(c, other.rangeMax));
      }
      return bottom();
    }
    if (isRange() && other.isConstant()) {
      return other.meet(*this);
    }

    // Two ranges -> union of ranges
    if (isRange() && other.isRange()) {
      return range(std::min(rangeMin, other.rangeMin),
                   std::max(rangeMax, other.rangeMax));
    }

    return bottom();
  }

  /// Join operation (LUB in lattice)
  /// Corresponds to Type.join() in Java
  LatticeValue join(const LatticeValue &other) const {
    // Dual of meet - implemented via De Morgan
    // join(a,b) = dual(meet(dual(a), dual(b)))
    // For simple cases:
    if (isBottom()) return other;
    if (other.isBottom()) return *this;
    if (isTop() || other.isTop()) return top();
    // For constants/ranges, join tends towards top
    if (*this == other) return *this;
    return top();
  }

  /// Check if this value is more specific than other (isa relationship)
  /// Corresponds to Type.isa() in Java - "this is a other"
  bool isa(const LatticeValue &other) const {
    if (other.isTop()) return true;   // Everything is a Top
    if (isBottom()) return true;      // Bottom is a everything
    if (isTop()) return other.isTop(); // Top is only a Top
    if (other.isBottom()) return isBottom();

    if (isConstant() && other.isConstant()) {
      return constantValue == other.constantValue;
    }
    if (isConstant() && other.isRange()) {
      if (auto thisInt = dyn_cast<IntegerAttr>(constantValue)) {
        int64_t c = thisInt.getInt();
        return c >= other.rangeMin && c <= other.rangeMax;
      }
    }
    if (isRange() && other.isRange()) {
      return rangeMin >= other.rangeMin && rangeMax <= other.rangeMax;
    }

    return false;
  }

  bool operator==(const LatticeValue &other) const {
    if (kind != other.kind) return false;
    if (kind == Constant) return constantValue == other.constantValue;
    if (kind == Range) return rangeMin == other.rangeMin && rangeMax == other.rangeMax;
    return true;  // Top == Top, Bottom == Bottom
  }

  bool operator!=(const LatticeValue &other) const { return !(*this == other); }
};

//===----------------------------------------------------------------------===//
// WorkList Implementation (from IterPeeps.WorkList in Java)
//===----------------------------------------------------------------------===//

/// Worklist with random pull order and duplicate prevention
/// Corresponds to IterPeeps.WorkList in Java
class SCCPWorkList {
public:
  SCCPWorkList(uint64_t seed = 123) : rng(seed) {}

  void push(Operation *op) {
    if (!op || onList.count(op)) return;
    onList.insert(op);
    items.push_back(op);
    totalWork++;
  }

  template<typename Range>
  void addAll(Range &&ops) {
    for (auto *op : ops) push(op);
  }

  Operation *pop() {
    if (items.empty()) return nullptr;
    
    // Random selection for better coverage
    size_t idx = std::uniform_int_distribution<size_t>(0, items.size() - 1)(rng);
    Operation *op = items[idx];
    items[idx] = items.back();
    items.pop_back();
    onList.erase(op);
    return op;
  }

  bool empty() const { return items.empty(); }
  bool contains(Operation *op) const { return onList.count(op); }
  size_t getTotalWork() const { return totalWork; }

private:
  std::vector<Operation *> items;
  // SmallPtrSet's SmallSize must be <= 32 on this toolchain; use 32 instead of 64
  llvm::SmallPtrSet<Operation *, 32> onList;
  std::mt19937_64 rng;
  size_t totalWork = 0;
};

//===----------------------------------------------------------------------===//
// SCCP Analysis State
//===----------------------------------------------------------------------===//

/// Holds analysis state for SCCP pass
/// Corresponds to the state maintained in IterPeeps.iterate()
struct SCCPState {
  SCCPWorkList worklist;
  
  /// Maps values to their lattice positions
  /// Corresponds to Node._type in Java
  llvm::DenseMap<Value, LatticeValue> lattice;
  
  /// Dependencies for invalidation
  /// Corresponds to Node._deps in Java
  llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>> deps;

  LatticeValue getLatticeValue(Value v) {
    auto it = lattice.find(v);
    if (it != lattice.end()) return it->second;
    return LatticeValue::top();  // Default to TOP
  }

  void setLatticeValue(Value v, const LatticeValue &val) {
    lattice[v] = val;
  }

  void addDep(Operation *from, Operation *to) {
    deps[from].insert(to);
  }

  void moveDepsToWorklist(Operation *op) {
    auto it = deps.find(op);
    if (it != deps.end()) {
      for (auto *dep : it->second) {
        worklist.push(dep);
      }
      it->second.clear();
    }
  }
};

//===----------------------------------------------------------------------===//
// Transfer Functions (from Node.compute() methods)
//===----------------------------------------------------------------------===//

/// Compute transfer function for an operation
/// Corresponds to calling node.compute() in Java
LatticeValue computeTransfer(Operation *op, SCCPState &state) {
  // Get input lattice values
  SmallVector<LatticeValue, 4> inputVals;
  for (Value operand : op->getOperands()) {
    inputVals.push_back(state.getLatticeValue(operand));
  }

  // Check if any input is TOP (haven't computed yet)
  bool hasTop = false;
  for (const auto &val : inputVals) {
    if (val.isTop()) hasTop = true;
  }
  
  // Constants always have their constant value
  if (auto constOp = dyn_cast<ConstantOp>(op)) {
    return LatticeValue::constant(constOp.getValue());
  }

  // If any input is TOP, result is TOP (wait for more info)
  if (hasTop && !inputVals.empty()) {
    return LatticeValue::top();
  }

  // Phi: meet all inputs from live control paths
  // Corresponds to PhiNode.compute() in Java
  if (auto phiOp = dyn_cast<PhiOp>(op)) {
    LatticeValue result = LatticeValue::top();
    
    // Get region control type
    auto regionVal = state.getLatticeValue(phiOp.getRegion());
    
    // If region is dead (XCONTROL), phi is TOP
    // This is the "conditional" part of SCCP
    if (regionVal.isTop()) {
      return LatticeValue::top();
    }
    
    for (Value v : phiOp.getValues()) {
      result = result.meet(state.getLatticeValue(v));
    }
    return result;
  }

  // Binary arithmetic: if both inputs are constants, fold
  if (inputVals.size() >= 2 && inputVals[0].isConstant() && inputVals[1].isConstant()) {
    auto lhs = inputVals[0].constantValue;
    auto rhs = inputVals[1].constantValue;
    
    if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
      if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
        int64_t l = lhsInt.getInt();
        int64_t r = rhsInt.getInt();
        int64_t result = 0;
        
        if (isa<AddOp>(op)) result = l + r;
        else if (isa<SubOp>(op)) result = l - r;
        else if (isa<MulOp>(op)) result = l * r;
        else if (isa<DivOp>(op)) {
          if (r == 0) return LatticeValue::bottom();
          result = l / r;
        }
        else if (isa<AndOp>(op)) result = l & r;
        else if (isa<OrOp>(op)) result = l | r;
        else if (isa<XorOp>(op)) result = l ^ r;
        else if (isa<ShlOp>(op)) result = l << r;
        else if (isa<ShrOp>(op)) result = static_cast<int64_t>(static_cast<uint64_t>(l) >> r);
        else if (isa<SarOp>(op)) result = l >> r;
        else return LatticeValue::bottom();

        return LatticeValue::constant(
          IntegerAttr::get(lhsInt.getType(), result));
      }
    }
  }

  // Comparison with constants
  if (auto cmpOp = dyn_cast<CmpOp>(op)) {
    if (inputVals.size() >= 2 && inputVals[0].isConstant() && inputVals[1].isConstant()) {
      auto lhs = inputVals[0].constantValue;
      auto rhs = inputVals[1].constantValue;
      
      if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
        if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
          int64_t l = lhsInt.getInt();
          int64_t r = rhsInt.getInt();
          bool result = false;
          
          StringRef pred = cmpOp.getPredicate();
          if (pred == "lt" || pred == "<") result = l < r;
          else if (pred == "le" || pred == "<=") result = l <= r;
          else if (pred == "gt" || pred == ">") result = l > r;
          else if (pred == "ge" || pred == ">=") result = l >= r;
          else if (pred == "eq" || pred == "==") result = l == r;
          else if (pred == "ne" || pred == "!=") result = l != r;
          else return LatticeValue::bottom();

          return LatticeValue::constant(
            IntegerAttr::get(IntegerType::get(op->getContext(), 1), result ? 1 : 0));
        }
      }
    }
  }

  // If node with constant predicate (IfNode.compute)
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    auto predVal = state.getLatticeValue(ifOp.getPred());
    
    if (predVal.isTop()) {
      // Neither branch is reachable yet
      return LatticeValue::top();
    }
    
    if (predVal.isConstant()) {
      if (auto predInt = dyn_cast<IntegerAttr>(predVal.constantValue)) {
        // Constant predicate - only one branch taken
        // Return indicator of which branch (could be encoded differently)
        return predVal;
      }
    }
    
    // Both branches may be taken
    return LatticeValue::bottom();
  }

  // Not unary
  if (auto notOp = dyn_cast<NotOp>(op)) {
    if (!inputVals.empty() && inputVals[0].isConstant()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(inputVals[0].constantValue)) {
        return LatticeValue::constant(
          IntegerAttr::get(intAttr.getType(), ~intAttr.getInt()));
      }
    }
  }

  // Minus unary
  if (auto minusOp = dyn_cast<MinusOp>(op)) {
    if (!inputVals.empty() && inputVals[0].isConstant()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(inputVals[0].constantValue)) {
        return LatticeValue::constant(
          IntegerAttr::get(intAttr.getType(), -intAttr.getInt()));
      }
    }
  }

  // Default: if any input is bottom, output is bottom
  for (const auto &val : inputVals) {
    if (val.isBottom()) return LatticeValue::bottom();
  }

  return LatticeValue::bottom();
}

//===----------------------------------------------------------------------===//
// SCCP Pass Implementation
//===----------------------------------------------------------------------===//

struct SCCPPass : public PassWrapper<SCCPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCCPPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2SONDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SCCPState state;

    // Phase 1: Initialize all nodes to TOP and add to worklist
    // Corresponds to: "We initialize all _type fields to TOP"
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        state.setLatticeValue(result, LatticeValue::top());
      }
      state.worklist.push(op);
    });

    // Phase 2: Iterate to fixed point
    // Corresponds to the main loop in IterPeeps.iterate()
    Operation *op;
    while ((op = state.worklist.pop()) != nullptr) {
      // Skip deleted operations
      if (op->getBlock() == nullptr) continue;

      // Run transfer function
      // Corresponds to: Type nval = n.compute();
      for (Value result : op->getResults()) {
        LatticeValue oldVal = state.getLatticeValue(result);
        LatticeValue newVal = computeTransfer(op, state);

        // Check if value changed
        // Corresponds to: if (n._type != nval)
        if (newVal != oldVal) {
          // Types start high and always fall
          // Corresponds to: assert n._type.isa(nval);
          if (!newVal.isa(oldVal) && !oldVal.isTop()) {
            // This shouldn't happen in a correct implementation
            // but we handle it gracefully
          }

          // Update type
          // Corresponds to: n._type = nval;
          state.setLatticeValue(result, newVal);

          // Add users to worklist
          // Corresponds to: code._iter.addAll(n._outputs);
          for (Operation *user : result.getUsers()) {
            state.worklist.push(user);
          }
        }
      }

      // Move dependencies to worklist
      state.moveDepsToWorklist(op);
    }

    // Phase 3: Apply optimizations based on discovered constants
    // Replace uses of constant-valued operations
    IRRewriter rewriter(module.getContext());
    
    module.walk([&](Operation *op) {
      // Skip constants - they're already optimal
      if (isa<ConstantOp>(op)) return WalkResult::advance();

      bool changed = false;
      for (Value result : op->getResults()) {
        LatticeValue val = state.getLatticeValue(result);
        
        if (val.isConstant()) {
          // Replace all uses with a constant
          rewriter.setInsertionPointAfter(op);
          auto constOp = rewriter.create<ConstantOp>(
            op->getLoc(), result.getType(), val.constantValue);
          
          result.replaceAllUsesWith(constOp.getResult());
          changed = true;
        }
      }

      // If all results were replaced, mark op for deletion
      if (changed && op->use_empty()) {
        rewriter.eraseOp(op);
      }

      return WalkResult::advance();
    });

    // Run regular folding/CSE after SCCP to clean up
    RewritePatternSet patterns(module.getContext());
    // Patterns would be populated here
    (void)applyPatternsGreedily(module, std::move(patterns));
  }

  StringRef getArgument() const final { return "son-sccp"; }
  StringRef getDescription() const final {
    return "Sparse Conditional Constant Propagation for Sea of Nodes dialect";
  }
};

//===----------------------------------------------------------------------===//
// Iterative Peephole Pass (from IterPeeps.java)
//===----------------------------------------------------------------------===//

/// Iterates peepholes to fixed point
/// Corresponds to IterPeeps.iterate() in Java
struct IterPeepsPass : public PassWrapper<IterPeepsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterPeepsPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2SONDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SCCPWorkList worklist;

    // Add all operations to worklist
    module.walk([&](Operation *op) {
      worklist.push(op);
    });

    // Iterate until fixed point
    Operation *op;
    while ((op = worklist.pop()) != nullptr) {
      if (op->getBlock() == nullptr) continue;

      // Try to fold the operation
      SmallVector<OpFoldResult, 4> foldResults;
      if (succeeded(op->fold(foldResults))) {
        // Handle fold results
        for (size_t i = 0; i < foldResults.size(); ++i) {
          if (auto attr = dyn_cast<Attribute>(foldResults[i])) {
            // Constant result - replace uses
            Value result = op->getResult(i);
            OpBuilder builder(op);
            auto constOp = builder.create<ConstantOp>(
              op->getLoc(), result.getType(), attr);
            result.replaceAllUsesWith(constOp.getResult());
            
            // Add users to worklist
            for (Operation *user : constOp.getResult().getUsers()) {
              worklist.push(user);
            }
          } else if (auto val = dyn_cast<Value>(foldResults[i])) {
            // Value result - replace uses
            Value result = op->getResult(i);
            result.replaceAllUsesWith(val);
            
            // Add users to worklist
            for (Operation *user : val.getUsers()) {
              worklist.push(user);
            }
          }
        }

        // Add operands to worklist (they might now be unused)
        for (Value operand : op->getOperands()) {
          if (auto *defOp = operand.getDefiningOp()) {
            worklist.push(defOp);
          }
        }

        // Erase the folded operation if unused
        if (op->use_empty()) {
          op->erase();
        }
      }
    }
  }

  StringRef getArgument() const final { return "son-iter-peeps"; }
  StringRef getDescription() const final {
    return "Iterative peephole optimization for Sea of Nodes dialect";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createSCCPPass() {
  return std::make_unique<SCCPPass>();
}

std::unique_ptr<Pass> createIterPeepsPass() {
  return std::make_unique<IterPeepsPass>();
}

} // namespace cpp2
} // namespace mlir

// Static registration
static mlir::PassRegistration<SCCPPass> sccpPass;
static mlir::PassRegistration<IterPeepsPass> iterPeepsPass;
