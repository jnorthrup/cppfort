//===- FIRTransferEliminationPass.cpp - Memory Transfer Elimination ------===//
///
/// Transfer Elimination pass for FIR dialect.
/// Uses escape analysis annotations to eliminate unnecessary GPU/DMA transfers
/// for variables that don't escape their local scope.
///
/// Phase 3 of Semantic AST Enhancements track.
///
//===----------------------------------------------------------------------===//

#include "Cpp2Passes.h"
#include "Cpp2FIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fir-transfer-elimination"

using namespace mlir;
using namespace mlir::cpp2fir;

namespace {

//===----------------------------------------------------------------------===//
// Escape Analysis Integration Types
//===----------------------------------------------------------------------===//

/// Escape kind for SSA values - mirrors the AST EscapeKind enum
enum class EscapeKind {
  NoEscape,          // Value stays local (stack) - can eliminate transfer
  EscapeToHeap,      // Stored in heap-allocated object
  EscapeToReturn,    // Returned from function
  EscapeToParam,     // Stored via pointer/reference parameter
  EscapeToGlobal,    // Stored in global variable
  EscapeToChannel,   // Sent through channel
  EscapeToGPU,       // Transferred to GPU memory - transfer required
  EscapeToDMA        // Transferred via DMA buffer - transfer required
};

/// Convert string attribute to EscapeKind
EscapeKind parseEscapeKind(StringRef str) {
  return llvm::StringSwitch<EscapeKind>(str)
      .Case("no_escape", EscapeKind::NoEscape)
      .Case("heap", EscapeKind::EscapeToHeap)
      .Case("return", EscapeKind::EscapeToReturn)
      .Case("param", EscapeKind::EscapeToParam)
      .Case("global", EscapeKind::EscapeToGlobal)
      .Case("channel", EscapeKind::EscapeToChannel)
      .Case("gpu", EscapeKind::EscapeToGPU)
      .Case("dma", EscapeKind::EscapeToDMA)
      .Default(EscapeKind::NoEscape);
}

/// Check if a transfer is required for this escape kind
bool transferRequired(EscapeKind kind) {
  switch (kind) {
    case EscapeKind::EscapeToGPU:
    case EscapeKind::EscapeToDMA:
      return true;
    default:
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Transfer Operation Analysis
//===----------------------------------------------------------------------===//

/// Represents a memory transfer operation that may be eliminated
struct TransferInfo {
  Operation *transferOp;      // The transfer operation
  Value source;               // Source value being transferred
  EscapeKind escapeKind;      // Escape kind of the source
  bool canEliminate;          // Whether this transfer can be eliminated
  std::string reason;         // Reason for elimination/retention
};

/// Analyze escape information for a value
/// Looks for #escape attribute on the defining operation
EscapeKind getEscapeKind(Value value) {
  if (!value)
    return EscapeKind::NoEscape;
  
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return EscapeKind::NoEscape;
  
  // Check for escape annotation attribute
  if (auto escapeAttr = defOp->getAttrOfType<StringAttr>("escape")) {
    return parseEscapeKind(escapeAttr.getValue());
  }
  
  // Default: conservative assumption based on operation type
  if (defOp->hasAttr("gpu_memory") || defOp->hasAttr("device_memory")) {
    return EscapeKind::EscapeToGPU;
  }
  if (defOp->hasAttr("dma_buffer")) {
    return EscapeKind::EscapeToDMA;
  }
  
  // Default to NoEscape for local values
  return EscapeKind::NoEscape;
}

/// Check if an operation is a transfer operation that can potentially be eliminated
bool isTransferOperation(Operation *op) {
  // Check operation name for transfer patterns
  StringRef opName = op->getName().getStringRef();
  
  return opName.contains("transfer") || 
         opName.contains("memcpy") ||
         opName.contains("dma") ||
         opName.contains("host_to_device") ||
         opName.contains("device_to_host");
}

//===----------------------------------------------------------------------===//
// Transfer Elimination Pass
//===----------------------------------------------------------------------===//

/// Transfer Elimination Pass for FIR Dialect
/// Eliminates unnecessary GPU/DMA transfers based on escape analysis
struct FIRTransferEliminationPass 
    : public PassWrapper<FIRTransferEliminationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRTransferEliminationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2FIRDialect>();
  }

  StringRef getArgument() const final { return "fir-transfer-elimination"; }
  StringRef getDescription() const final {
    return "Eliminate unnecessary GPU/DMA transfers based on escape analysis";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "=== FIR Transfer Elimination Pass Start ===\n");

    // Statistics for optimization reporting
    unsigned totalTransfers = 0;
    unsigned eliminatedTransfers = 0;
    unsigned keptTransfers = 0;
    
    // Collect transfer operations and their escape information
    SmallVector<TransferInfo, 16> transfers;
    
    // Phase 1: Identify all transfer operations
    LLVM_DEBUG(llvm::dbgs() << "\n=== Phase 1: Identifying Transfers ===\n");
    
    module.walk([&](Operation *op) {
      if (isTransferOperation(op)) {
        totalTransfers++;
        TransferInfo info;
        info.transferOp = op;
        
        // Get the source value (typically the first operand for transfer ops)
        if (op->getNumOperands() > 0) {
          info.source = op->getOperand(0);
          info.escapeKind = getEscapeKind(info.source);
        } else {
          info.escapeKind = EscapeKind::NoEscape;
        }
        
        // Determine if transfer can be eliminated
        info.canEliminate = !transferRequired(info.escapeKind);
        
        if (info.canEliminate) {
          info.reason = "Source value does not escape to GPU/DMA";
        } else {
          info.reason = "Source value escapes to external memory";
        }
        
        LLVM_DEBUG({
          llvm::dbgs() << "  Transfer: " << op->getName() << "\n";
          llvm::dbgs() << "    Can eliminate: " << (info.canEliminate ? "yes" : "no") << "\n";
          llvm::dbgs() << "    Reason: " << info.reason << "\n";
        });
        
        transfers.push_back(std::move(info));
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Found " << totalTransfers << " transfer operations\n");

    // Phase 2: Validate elimination candidates
    // Ensure no downstream operations depend on the eliminated transfer
    LLVM_DEBUG(llvm::dbgs() << "\n=== Phase 2: Validating Eliminations ===\n");
    
    for (auto &info : transfers) {
      if (!info.canEliminate)
        continue;
        
      // Check if any users require the transfer to happen
      for (auto result : info.transferOp->getResults()) {
        for (auto *user : result.getUsers()) {
          // If a user has GPU/DMA attributes, we cannot eliminate
          if (user->hasAttr("gpu_memory") || user->hasAttr("device_memory") ||
              user->hasAttr("requires_transfer")) {
            info.canEliminate = false;
            info.reason = "Downstream operation requires transferred value";
            LLVM_DEBUG({
              llvm::dbgs() << "  Cannot eliminate: downstream user requires transfer\n";
            });
            break;
          }
        }
      }
    }

    // Phase 3: Apply eliminations
    LLVM_DEBUG(llvm::dbgs() << "\n=== Phase 3: Applying Eliminations ===\n");
    
    for (auto &info : transfers) {
      if (info.canEliminate) {
        LLVM_DEBUG({
          llvm::dbgs() << "  Eliminating: " << info.transferOp->getName() << "\n";
          llvm::dbgs() << "    Reason: " << info.reason << "\n";
        });
        
        // For transfers with results, we need to reroute uses to the source
        if (info.transferOp->getNumResults() > 0 && info.source) {
          // Replace uses of transfer result with the source value directly
          // This is the key optimization: bypassing the unnecessary transfer
          info.transferOp->getResult(0).replaceAllUsesWith(info.source);
        }
        
        // Erase the transfer operation
        info.transferOp->erase();
        eliminatedTransfers++;
      } else {
        keptTransfers++;
        LLVM_DEBUG({
          llvm::dbgs() << "  Keeping: " << info.transferOp->getName() << "\n";
          llvm::dbgs() << "    Reason: " << info.reason << "\n";
        });
      }
    }

    // Report statistics
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Transfer Elimination Summary ===\n";
      llvm::dbgs() << "  Total transfers: " << totalTransfers << "\n";
      llvm::dbgs() << "  Eliminated: " << eliminatedTransfers << "\n";
      llvm::dbgs() << "  Kept: " << keptTransfers << "\n";
      if (totalTransfers > 0) {
        double percentage = (double)eliminatedTransfers / totalTransfers * 100.0;
        llvm::dbgs() << "  Reduction: " << llvm::format("%.1f%%", percentage) << "\n";
      }
      llvm::dbgs() << "=== FIR Transfer Elimination Pass End ===\n\n";
    });
    
    // Mark module as modified if we eliminated any transfers
    if (eliminatedTransfers > 0) {
      // Module was modified - no need to explicitly mark in MLIR
    }
  }
};

//===----------------------------------------------------------------------===//
// DMA Safety Validation Pass
//===----------------------------------------------------------------------===//

/// DMA Safety Validation Pass
/// Validates that no aliasing occurs during async DMA transfers
struct FIRDMASafetyPass 
    : public PassWrapper<FIRDMASafetyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRDMASafetyPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2FIRDialect>();
  }

  StringRef getArgument() const final { return "fir-dma-safety"; }
  StringRef getDescription() const final {
    return "Validate DMA safety rules (no aliasing during async transfers)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool hasErrors = false;
    
    LLVM_DEBUG(llvm::dbgs() << "=== FIR DMA Safety Validation Start ===\n");

    // Track active DMA transfers and their memory regions
    DenseMap<Value, Operation*> activeTransfers;
    
    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      
      // Check for DMA start operations
      if (opName.contains("dma_transfer") || opName.contains("async_copy")) {
        if (op->getNumOperands() >= 2) {
          Value src = op->getOperand(0);
          Value dst = op->getOperand(1);
          
          // Check for aliasing with active transfers
          if (activeTransfers.count(src) || activeTransfers.count(dst)) {
            op->emitError("Potential aliasing during active DMA transfer");
            hasErrors = true;
          }
          
          // Track this transfer
          for (Value result : op->getResults()) {
            activeTransfers[result] = op;
          }
        }
      }
      
      // Check for DMA completion/wait operations
      if (opName.contains("dma_wait") || opName.contains("sync")) {
        // DMA completed - can clear tracking for this region
        for (Value operand : op->getOperands()) {
          activeTransfers.erase(operand);
        }
      }
    });

    LLVM_DEBUG({
      llvm::dbgs() << "DMA safety validation " 
                   << (hasErrors ? "FAILED" : "PASSED") << "\n";
      llvm::dbgs() << "=== FIR DMA Safety Validation End ===\n\n";
    });

    if (hasErrors) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createFIRTransferEliminationPass() {
  return std::make_unique<FIRTransferEliminationPass>();
}

std::unique_ptr<Pass> createFIRDMASafetyPass() {
  return std::make_unique<FIRDMASafetyPass>();
}

} // namespace cpp2
} // namespace mlir

// Static registration
static mlir::PassRegistration<FIRTransferEliminationPass> firTransferEliminationPass;
static mlir::PassRegistration<FIRDMASafetyPass> firDmaSafetyPass;
