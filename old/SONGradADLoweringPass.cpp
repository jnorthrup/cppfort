//===- SONGradADLoweringPass.cpp - Gradient AD Lowering Pass -------------===//
//
// Lower gradient protocol operations (grad_diff) to arithmetic operations.
// Implements forward-mode automatic differentiation using the chain rule.
//
//===----------------------------------------------------------------------===//

#include "Cpp2Passes.h"
#include "Cpp2SONDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sond;

namespace {

//===----------------------------------------------------------------------===//
// Gradient AD Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower grad_diff on multiplication: d/dx(f * g) = f' * g + f * g'
struct LowerGradDiffMul : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    // Check if expression is a multiplication
    auto mulOp = expr.getDefiningOp<MulFOp>();
    if (!mulOp)
      mulOp = expr.getDefiningOp<MulOp>();
    
    if (!mulOp)
      return failure();

    auto lhs = mulOp.getLhs();
    auto rhs = mulOp.getRhs();

    // Create grad_diff for lhs and rhs
    auto gradLhs = rewriter.create<GradDiffOp>(op.getLoc(), lhs, var);
    auto gradRhs = rewriter.create<GradDiffOp>(op.getLoc(), rhs, var);

    // Create: f' * g + f * g'
    auto term1 = rewriter.create<MulFOp>(op.getLoc(), gradLhs.getResult(), rhs);
    auto term2 = rewriter.create<MulFOp>(op.getLoc(), lhs, gradRhs.getResult());
    auto result = rewriter.create<AddFOp>(op.getLoc(), term1.getResult(), term2.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower grad_diff on addition: d/dx(f + g) = f' + g'
struct LowerGradDiffAdd : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    auto addOp = expr.getDefiningOp<AddFOp>();
    if (!addOp)
      addOp = expr.getDefiningOp<AddOp>();
    
    if (!addOp)
      return failure();

    auto lhs = addOp.getLhs();
    auto rhs = addOp.getRhs();

    auto gradLhs = rewriter.create<GradDiffOp>(op.getLoc(), lhs, var);
    auto gradRhs = rewriter.create<GradDiffOp>(op.getLoc(), rhs, var);
    auto result = rewriter.create<AddFOp>(op.getLoc(), gradLhs.getResult(), gradRhs.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower grad_diff on subtraction: d/dx(f - g) = f' - g'
struct LowerGradDiffSub : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    auto subOp = expr.getDefiningOp<SubFOp>();
    if (!subOp)
      subOp = expr.getDefiningOp<SubOp>();
    
    if (!subOp)
      return failure();

    auto lhs = subOp.getLhs();
    auto rhs = subOp.getRhs();

    auto gradLhs = rewriter.create<GradDiffOp>(op.getLoc(), lhs, var);
    auto gradRhs = rewriter.create<GradDiffOp>(op.getLoc(), rhs, var);
    auto result = rewriter.create<SubFOp>(op.getLoc(), gradLhs.getResult(), gradRhs.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower grad_diff on division: d/dx(f / g) = (f' * g - f * g') / g^2
struct LowerGradDiffDiv : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    auto divOp = expr.getDefiningOp<DivFOp>();
    if (!divOp)
      divOp = expr.getDefiningOp<DivOp>();
    
    if (!divOp)
      return failure();

    auto lhs = divOp.getLhs();
    auto rhs = divOp.getRhs();

    // grad(f/g) = (f' * g - f * g') / g^2
    auto gradLhs = rewriter.create<GradDiffOp>(op.getLoc(), lhs, var);
    auto gradRhs = rewriter.create<GradDiffOp>(op.getLoc(), rhs, var);

    auto term1 = rewriter.create<MulFOp>(op.getLoc(), gradLhs.getResult(), rhs);
    auto term2 = rewriter.create<MulFOp>(op.getLoc(), lhs, gradRhs.getResult());
    auto numerator = rewriter.create<SubFOp>(op.getLoc(), term1.getResult(), term2.getResult());

    auto gSquared = rewriter.create<MulFOp>(op.getLoc(), rhs, rhs);
    auto result = rewriter.create<DivFOp>(op.getLoc(), numerator.getResult(), gSquared.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower grad_diff on a constant: d/dx(c) = 0
struct LowerGradDiffConstant : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();

    // If expression is a constant, gradient is zero
    if (expr.getDefiningOp<ConstantOp>()) {
      auto resultType = op.getResult().getType();
      auto zeroAttr = rewriter.getZeroAttr(resultType);
      auto zero = rewriter.create<ConstantOp>(op.getLoc(), resultType, zeroAttr);
      rewriter.replaceOp(op, zero.getResult());
      return success();
    }

    return failure();
  }
};

/// Lower grad_diff when differentiating w.r.t. the variable itself: d/dx(x) = 1
struct LowerGradDiffSelf : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    // If differentiating variable with respect to itself: d/dx(x) = 1
    if (expr == var) {
      auto resultType = op.getResult().getType();
      auto oneAttr = rewriter.getIntegerAttr(resultType, 1);
      auto one = rewriter.create<ConstantOp>(op.getLoc(), resultType, oneAttr);
      rewriter.replaceOp(op, one.getResult());
      return success();
    }

    return failure();
  }
};

/// Lower grad_diff on indexed/series: chain rule through the callable
struct LowerGradDiffIndexed : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    auto indexedOp = expr.getDefiningOp<IndexedOp>();
    if (!indexedOp)
      indexedOp = expr.getDefiningOp<SeriesOp>();
    
    if (!indexedOp)
      return failure();

    // grad(indexed) = jacobian(indexed) * grad(var)
    // For a simple indexed, this is the derivative of the 'at' function
    auto jacobian = rewriter.create<JacobianOp>(op.getLoc(), indexedOp.getAt(), var);
    auto result = rewriter.create<MatrixMulOp>(op.getLoc(), jacobian.getResult(), var);

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower grad_diff on atlas/manifold: manifold chain rule
struct LowerGradDiffAtlas : public OpRewritePattern<GradDiffOp> {
  using OpRewritePattern<GradDiffOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GradDiffOp op, PatternRewriter &rewriter) const override {
    auto expr = op.getExpr();
    auto var = op.getVariable();

    auto atlasOp = expr.getDefiningOp<AtlasOp>();
    if (!atlasOp)
      atlasOp = expr.getDefiningOp<ManifoldOp>();
    
    if (!atlasOp)
      return failure();

    // grad(atlas) = jacobian(charts) * inner_grad
    // The manifold chain rule: gradient is Jacobian times inner gradient
    auto jacobian = rewriter.create<JacobianOp>(op.getLoc(), atlasOp.getCharts(), var);
    auto innerGrad = rewriter.create<GradDiffOp>(op.getLoc(), atlasOp.getCharts(), var);
    auto result = rewriter.create<MatrixMulOp>(op.getLoc(), jacobian.getResult(), innerGrad.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GradAD Lowering Pass
//===----------------------------------------------------------------------===//

struct SONGradADLoweringPass
    : public PassWrapper<SONGradADLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SONGradADLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2SONDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());

    // Add all lowering patterns
    patterns.add<LowerGradDiffMul>(&getContext());
    patterns.add<LowerGradDiffAdd>(&getContext());
    patterns.add<LowerGradDiffSub>(&getContext());
    patterns.add<LowerGradDiffDiv>(&getContext());
    patterns.add<LowerGradDiffConstant>(&getContext());
    patterns.add<LowerGradDiffSelf>(&getContext());
    patterns.add<LowerGradDiffIndexed>(&getContext());
    patterns.add<LowerGradDiffAtlas>(&getContext());

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "son-grad-ad-lowering"; }
  StringRef getDescription() const final {
    return "Lower gradient protocol to arithmetic operations";
  }
};

} // namespace

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createSONGradADLoweringPass() {
  return std::make_unique<SONGradADLoweringPass>();
}

} // namespace cpp2
} // namespace mlir

// Static pass registration
static mlir::PassRegistration<SONGradADLoweringPass> gradADPass;
