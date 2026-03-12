// JMM Constraint Verification for Cpp2 SON Dialect
// Validates Java Memory Model constraints in Sea of Nodes operations
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Visitors.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "Cpp2SONDialect.h"

using namespace mlir;
using namespace mlir::sond;

//===----------------------------------------------------------------------===//
// JMM Constraint Verification
//===----------------------------------------------------------------------===//

namespace {

/// JMM constraint violations collected during verification
enum class JMMViolationKind {
  /// Happens-before edge references non-existent operation
  MissingPredecessor,
  /// Happens-before edge forms a cycle
  CyclicDependency,
  /// Volatile operation lacks shared visibility
  VolatileWithoutShared,
  /// Final field not frozen in constructor
  FinalFieldNotFrozen,
  /// Final field frozen after constructor exit
  FinalFieldFrozenLate,
  /// Unsafe publication: object visible before final fields frozen
  UnsafePublication,
  /// Concurrent operation without happens-before edges
  MissingSyncEdge
};

struct JMMViolation {
  JMMViolationKind kind;
  Operation *op;
  std::string message;

  JMMViolation(JMMViolationKind kind, Operation *op, const std::string &message)
      : kind(kind), op(op), message(message) {}
};

/// JMM constraint verifier for SON operations
class JMMConstraintVerifier {
public:
  explicit JMMConstraintVerifier(Operation *rootOp) : rootOp(rootOp) {}

  /// Run verification and collect violations
  LogicalResult verify() {
    violations.clear();

    // Walk all operations and verify JMM constraints
    if (failed(walkAndVerify())) {
      return failure();
    }

    // Analyze happens-before graph for cycles
    if (failed(verifyHappensBeforeConsistency())) {
      return failure();
    }

    return success();
  }

  /// Get collected violations
  const SmallVector<JMMViolation> &getViolations() const { return violations; }

  /// Print violations to output stream
  void printViolations(raw_ostream &os) const {
    if (violations.empty()) {
      os << "No JMM violations found.\n";
      return;
    }

    os << "JMM Constraint Violations (" << violations.size() << "):\n";
    for (const auto &violation : violations) {
      os << "  - ";
      switch (violation.kind) {
      case JMMViolationKind::MissingPredecessor:
        os << "MissingPredecessor";
        break;
      case JMMViolationKind::CyclicDependency:
        os << "CyclicDependency";
        break;
      case JMMViolationKind::VolatileWithoutShared:
        os << "VolatileWithoutShared";
        break;
      case JMMViolationKind::FinalFieldNotFrozen:
        os << "FinalFieldNotFrozen";
        break;
      case JMMViolationKind::FinalFieldFrozenLate:
        os << "FinalFieldFrozenLate";
        break;
      case JMMViolationKind::UnsafePublication:
        os << "UnsafePublication";
        break;
      case JMMViolationKind::MissingSyncEdge:
        os << "MissingSyncEdge";
        break;
      }
      os << ": " << violation.message << "\n";

      if (violation.op) {
        os << "    at: ";
        violation.op->print(os);
        os << "\n";
      }
    }
  }

private:
  Operation *rootOp;
  SmallVector<JMMViolation> violations;
  DenseMap<StringRef, Operation*> opNames;

  /// Walk all operations and verify JMM constraints
  LogicalResult walkAndVerify() {
    return rootOp->walk([&](Operation *op) {
      // Collect named operations for happens-before resolution
      if (auto nameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
        opNames[nameAttr.getValue()] = op;
      }

      // Verify LoadOp JMM constraints
      if (auto loadOp = dyn_cast<LoadOp>(op)) {
        if (failed(verifyLoadOp(loadOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify StoreOp JMM constraints
      if (auto storeOp = dyn_cast<StoreOp>(op)) {
        if (failed(verifyStoreOp(storeOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify NewOp JMM constraints
      if (auto newOp = dyn_cast<NewOp>(op)) {
        if (failed(verifyNewOp(newOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify SendOp JMM constraints
      if (auto sendOp = dyn_cast<SendOp>(op)) {
        if (failed(verifySendOp(sendOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify RecvOp JMM constraints
      if (auto recvOp = dyn_cast<RecvOp>(op)) {
        if (failed(verifyRecvOp(recvOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify SpawnOp JMM constraints
      if (auto spawnOp = dyn_cast<SpawnOp>(op)) {
        if (failed(verifySpawnOp(spawnOp))) {
          return WalkResult::interrupt();
        }
      }

      // Verify AwaitOp JMM constraints
      if (auto awaitOp = dyn_cast<AwaitOp>(op)) {
        if (failed(verifyAwaitOp(awaitOp))) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    }).wasInterrupted() ? failure() : success();
  }

  /// Verify LoadOp JMM constraints
  LogicalResult verifyLoadOp(LoadOp loadOp) {
    // If volatile, must have shared visibility
    if (loadOp.getJmmVolatile()) {
      auto visibilityAttr = loadOp.getJmmVisibility();
      if (!visibilityAttr || visibilityAttr->getVisibilityKind() != "shared") {
        violations.push_back(JMMViolation(
            JMMViolationKind::VolatileWithoutShared,
            loadOp.getOperation(),
            "volatile load must have shared visibility"));
      }
    }

    // Verify happens-before predecessors exist
    if (auto hbAttr = loadOp.getJmmHappensBefore()) {
      for (StringRef pred : hbAttr->getPredecessors()) {
        if (!opNames.count(pred)) {
          violations.push_back(JMMViolation(
              JMMViolationKind::MissingPredecessor,
              loadOp.getOperation(),
              "happens-before predecessor not found: " + pred.str()));
        }
      }
    }

    return success();
  }

  /// Verify StoreOp JMM constraints
  LogicalResult verifyStoreOp(StoreOp storeOp) {
    // If volatile, must have shared visibility
    if (storeOp.getJmmVolatile()) {
      auto visibilityAttr = storeOp.getJmmVisibility();
      if (!visibilityAttr || visibilityAttr->getVisibilityKind() != "shared") {
        violations.push_back(JMMViolation(
            JMMViolationKind::VolatileWithoutShared,
            storeOp.getOperation(),
            "volatile store must have shared visibility"));
      }
    }

    // Verify final field is frozen during construction
    if (auto finalAttr = storeOp.getJmmFinalField()) {
      if (!finalAttr->getIsFrozen()) {
        violations.push_back(JMMViolation(
            JMMViolationKind::FinalFieldNotFrozen,
            storeOp.getOperation(),
            "final field must be frozen in constructor"));
      }
    }

    // Verify happens-before predecessors exist
    if (auto hbAttr = storeOp.getJmmHappensBefore()) {
      for (StringRef pred : hbAttr->getPredecessors()) {
        if (!opNames.count(pred)) {
          violations.push_back(JMMViolation(
              JMMViolationKind::MissingPredecessor,
              storeOp.getOperation(),
              "happens-before predecessor not found: " + pred.str()));
        }
      }
    }

    return success();
  }

  /// Verify NewOp JMM constraints (constructor boundary)
  LogicalResult verifyNewOp(NewOp newOp) {
    // If constructor, check for final field safety
    if (newOp.getIsConstructor()) {
      // Constructor must establish shared visibility for object
      auto visibilityAttr = newOp.getJmmVisibility();
      if (visibilityAttr && visibilityAttr->getVisibilityKind() != "shared") {
        violations.push_back(JMMViolation(
            JMMViolationKind::UnsafePublication,
            newOp.getOperation(),
            "constructor should produce shared-visible object"));
      }
    }

    return success();
  }

  /// Verify SendOp JMM constraints
  LogicalResult verifySendOp(SendOp sendOp) {
    // Send must establish happens-before with matching recv
    if (!sendOp.getJmmHappensBefore()) {
      violations.push_back(JMMViolation(
          JMMViolationKind::MissingSyncEdge,
          sendOp.getOperation(),
          "send operation must have happens-before edge to recv"));
    }

    return success();
  }

  /// Verify RecvOp JMM constraints
  LogicalResult verifyRecvOp(RecvOp recvOp) {
    // Recv must establish happens-before with matching send
    if (!recvOp.getJmmHappensBefore()) {
      violations.push_back(JMMViolation(
          JMMViolationKind::MissingSyncEdge,
          recvOp.getOperation(),
          "recv operation must have happens-before edge from send"));
    }

    return success();
  }

  /// Verify SpawnOp JMM constraints
  LogicalResult verifySpawnOp(SpawnOp spawnOp) {
    // Spawn must establish happens-before with child start
    if (!spawnOp.getJmmHappensBefore()) {
      violations.push_back(JMMViolation(
          JMMViolationKind::MissingSyncEdge,
          spawnOp.getOperation(),
          "spawn operation must have happens-before edge to spawned thread"));
    }

    return success();
  }

  /// Verify AwaitOp JMM constraints
  LogicalResult verifyAwaitOp(AwaitOp awaitOp) {
    // Await must establish happens-before with child completion
    if (!awaitOp.getJmmHappensBefore()) {
      violations.push_back(JMMViolation(
          JMMViolationKind::MissingSyncEdge,
          awaitOp.getOperation(),
          "await operation must have happens-before edge from thread completion"));
    }

    return success();
  }

  /// Verify happens-before graph for cycles and consistency
  LogicalResult verifyHappensBeforeConsistency() {
    // Build adjacency list for happens-before graph
    DenseMap<Operation*, SmallVector<Operation*>> graph;

    // Collect all happens-before edges
    for (auto &[name, op] : opNames) {
      if (auto hbAttr = op->getAttrOfType<JMMHappensBeforeAttr>("jmm_happens_before")) {
        for (StringRef predName : hbAttr.getPredecessors()) {
          if (auto *predOp = opNames.lookup(predName)) {
            graph[predOp].push_back(op);
          }
        }
      }
    }

    // Detect cycles using DFS
    DenseSet<Operation*> visiting;
    DenseSet<Operation*> visited;

    std::function<bool(Operation*)> dfs = [&](Operation *node) -> bool {
      if (visited.contains(node)) return false;
      if (visiting.contains(node)) {
        violations.push_back(JMMViolation(
            JMMViolationKind::CyclicDependency,
            node,
            "cycle detected in happens-before graph"));
        return true;
      }

      visiting.insert(node);
      for (Operation *succ : graph.lookup(node)) {
        if (dfs(succ)) return true;
      }
      visiting.erase(node);
      visited.insert(node);
      return false;
    };

    for (auto &[name, op] : opNames) {
      if (dfs(op)) {
        return failure();
      }
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sond {

/// Verify JMM constraints in a SON operation
LogicalResult verifyJMMConstraints(Operation *op) {
  JMMConstraintVerifier verifier(op);
  if (failed(verifier.verify())) {
    verifier.printViolations(llvm::errs());
    return failure();
  }
  return success();
}

/// Verify JMM constraints in a module
bool verifyJMMConstraints(ModuleOp module) {
  return succeeded(verifyJMMConstraints(module.getOperation()));
}

} // namespace sond
} // namespace mlir
