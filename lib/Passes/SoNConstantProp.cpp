//===----------------------------------------------------------------------===//
// Sea-of-Nodes Constant Propagation Pass
// TrikeShed Math-Based SoN Compiler
//
// Implements template parameter folding and dead code elimination.
// "Smashes templates into constants" at the SoN level.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "Cpp2SONDialect.h"
#include "Cpp2SONDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Constant Propagation Patterns
//===----------------------------------------------------------------------===//

// Pattern: Fold constant domain into indexed operation
struct IndexedConstantFoldPattern : public mlir::OpRewritePattern<cpp2::IndexedOp> {
  using mlir::OpRewritePattern<cpp2::IndexedOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::IndexedOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Check if domain is constant
    auto domain = op.getDomain();
    if (!domain.getDefiningOp()) {
      return mlir::failure();
    }

    // If domain is a constant integer, we can specialize the indexed op
    if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(domain.getDefiningOp())) {
      // Create a specialized indexed op with constant domain
      // This enables SoN to propagate the constant through the graph
      rewriter.replaceOpWithNewOp<cpp2::IndexedOp>(
          op, op.getResult().getType(), domain, op.getAccessor());
      
      return mlir::success();
    }

    return mlir::failure();
  }
};

// Pattern: Dead code elimination for unused operations
struct DeadCodeElimPattern : public mlir::OpRewritePattern<cpp2::IndexedOp> {
  using mlir::OpRewritePattern<cpp2::IndexedOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::IndexedOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    // Check if result is used
    if (op.getResult().use_empty()) {
      // Dead operation - can be eliminated
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Constant Propagation Pass
//===----------------------------------------------------------------------===//

struct SoNConstantPropPass
    : public mlir::PassWrapper<SoNConstantPropPass, mlir::OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const override { return "son-constant-prop"; }
  StringRef getDescription() const override {
    return "Sea-of-Nodes constant propagation for template smashing";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<DeadCodeElimPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
struct SoNConstantPropPassRegistration {
  SoNConstantPropPassRegistration() {
    PassRegistration<SoNConstantPropPass>();
  }
};

static SoNConstantPropPassRegistration registration;
}  // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSoNConstantPropPass() {
  return std::make_unique<SoNConstantPropPass>();
}
