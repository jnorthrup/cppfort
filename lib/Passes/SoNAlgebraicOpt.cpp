//===----------------------------------------------------------------------===//
// Sea-of-Nodes Algebraic Optimizer Pass
// TrikeShed Math-Based SoN Compiler
//
// Implements algebraic rewrite rules for the Sea-of-Nodes dialect:
// - Join associativity, commutativity, identity
// - Transition idempotence
// - Chart project/embed inverses
// - Indexed composition simplifications
//
// Mathematical Basis:
// The Sea-of-Nodes IR forms a categorical structure where:
// - Join forms a monoidal category (associative, commutative, identity)
// - Charts form a smooth manifold structure
// - Indexed operations form function spaces
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Cpp2SONDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Algebraic Rewrite Rules
//===----------------------------------------------------------------------===//

namespace {

//--------------------------------------------------------------------------//
// Rule 1: Join Associativity
// Pattern: (a j b) j c => a j (b j c)
//
// Mathematical Proof:
// In a categorical product, the binary join (product) is associative:
// Given objects A, B, C with projections p1: A×B→A, p2: A×B→B
// and q1: B×C→B, q2: B×C→C, there exists a unique object (A×B)×C
// isomorphic to A×(B×C). This is the universal property of the product.
// Therefore: (A ⊔ B) ⊔ C ≅ A ⊔ (B ⊔ C)
//--------------------------------------------------------------------------//
struct JoinAssociativityPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if LHS is itself a JoinOp (nested join)
    auto lhsJoin = lhs.getDefiningOp<cpp2::JoinOp>();
    if (!lhsJoin) {
      return mlir::failure();
    }

    // Transform: (a j b) j c => a j (b j c)
    auto newInnerJoin = rewriter.create<cpp2::JoinOp>(
        op.getLoc(), lhsJoin.getRhs(), rhs);
    auto newOuterJoin = rewriter.create<cpp2::JoinOp>(
        op.getLoc(), lhsJoin.getLhs(), newInnerJoin.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), newOuterJoin.getResult());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//--------------------------------------------------------------------------//
// Rule 2: Join Commutativity
// Pattern: a j b => b j a
//
// Mathematical Proof:
// The categorical product is symmetric: A×B ≅ B×A
// This follows from the universal property - the projections can be swapped.
//--------------------------------------------------------------------------//
struct JoinCommutativityPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Apply only when beneficial (e.g., canonical ordering)
    // Prefer: constant on right, variable on left
    if (isConstantLike(lhs) && !isConstantLike(rhs)) {
      auto newJoin = rewriter.create<cpp2::JoinOp>(
          op.getLoc(), rhs, lhs);
      rewriter.replaceAllUsesWith(op.getResult(), newJoin.getResult());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }

private:
  bool isConstantLike(mlir::Value val) const {
    if (auto op = val.getDefiningOp()) {
      return isa<mlir::arith::ConstantOp, mlir::arith::ConstantIntOp,
                 mlir::arith::ConstantFloatOp>(op);
    }
    return false;
  }
};

//--------------------------------------------------------------------------//
// Rule 3: Transition Idempotence
// Pattern: transition(m, c, c, coords) => coords
//
// Mathematical Proof:
// In manifold theory, transition between identical charts is identity:
// For chart φ_c: M → ℝⁿ, we have φ_c ∘ φ_c⁻¹ = id
// Therefore: transition(M, c, c, p) = p
//--------------------------------------------------------------------------//
struct TransitionIdempotencePattern : public mlir::OpRewritePattern<cpp2::TransitionOp> {
  using mlir::OpRewritePattern<cpp2::TransitionOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::TransitionOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto fromChart = op.getFromChart();
    auto toChart = op.getToChart();

    // Check if from_chart == to_chart (same chart transition)
    if (fromChart == toChart) {
      // Transition to same chart is identity
      rewriter.replaceAllUsesWith(op.getResult(), op.getCoords());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//--------------------------------------------------------------------------//
// Rule 4: Chart Project/Embed Inverse
// Pattern: chart_project(chart_embed(chart, local), chart) => local
//
// Mathematical Proof:
// Chart embedding followed by projection returns the original coordinates:
// φ ∘ φ⁻¹ = id by definition of chart inverse
//--------------------------------------------------------------------------//
struct ChartInversePattern : public mlir::OpRewritePattern<cpp2::ChartProjectOp> {
  using mlir::OpRewritePattern<cpp2::ChartProjectOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::ChartProjectOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto point = op.getPoint();
    auto chart = op.getChart();

    // Check if point comes from chart_embed with same chart
    auto embedOp = point.getDefiningOp<cpp2::ChartEmbedOp>();
    if (!embedOp) {
      return mlir::failure();
    }

    // Verify same chart
    if (embedOp.getChart() != chart) {
      return mlir::failure();
    }

    // chart_project(chart_embed(chart, local), chart) = local
    rewriter.replaceAllUsesWith(op.getResult(), embedOp.getLocalCoords());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//--------------------------------------------------------------------------//
// Rule 5: Indexed Identity
// Pattern: indexed(domain, identity_accessor) => domain
//
// Mathematical Proof:
// If the accessor is identity (id(x) = x), then indexed<I, id> = I
//--------------------------------------------------------------------------//
struct IndexedIdentityPattern : public mlir::OpRewritePattern<cpp2::IndexedOp> {
  using mlir::OpRewritePattern<cpp2::IndexedOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::IndexedOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto accessor = op.getAccessor();

    // Check if accessor is identity function (typically a constant or known identity)
    // This would require analysis of the accessor function
    // For now, handle the case where domain type matches result type (id case)
    if (op.getResult().getType() == op.getDomain().getType() &&
        accessor.getType().isa<mlir::NoneType>()) {
      // Identity indexing: indexed<I, λx.x>(x) = x
      rewriter.replaceAllUsesWith(op.getResult(), op.getDomain());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//--------------------------------------------------------------------------//
// Rule 6: Dead Join Elimination
// Pattern: join(x, unused) => x when unused has no other uses
//
// Mathematical Proof:
// In a monoidal category with unit (), we have:
// a ⊔ () ≅ a (unit is identity)
// If one component is unused and has no other dependencies, it can be dropped.
//--------------------------------------------------------------------------//
struct DeadJoinElimPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if RHS is unused and can be eliminated
    if (rhs.getUses().empty() && isDroppable(rhs)) {
      rewriter.replaceAllUsesWith(op.getResult(), lhs);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Check if LHS is unused
    if (lhs.getUses().empty() && isDroppable(lhs)) {
      rewriter.replaceAllUsesWith(op.getResult(), rhs);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool isDroppable(mlir::Value val) const {
    if (auto op = val.getDefiningOp()) {
      // Only drop trivial operations (constants, coordinates)
      return isa<cpp2::CoordinatesOp, mlir::arith::ConstantOp,
                 mlir::arith::ConstantIntOp, mlir::arith::ConstantFloatOp>(op);
    }
    return false;
  }
};

//--------------------------------------------------------------------------//
// Rule 7: Coordinates Canonicalization
// Pattern: coords[c1, c2, ...] with repeated values => coords[canonical]
//
// Mathematical Proof:
// Coordinate tuples with duplicate values can be reduced.
// E.g., coords[1.0, 1.0, x] ≅ coords[1.0, x] (idempotence)
//--------------------------------------------------------------------------//
struct CoordinatesCanonicalPattern : public mlir::OpRewritePattern<cpp2::CoordinatesOp> {
  using mlir::OpRewritePattern<cpp2::CoordinatesOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::CoordinatesOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    // This would require analyzing the values array for duplicates
    // For now, just handle simple constant case where all values are same
    // In practice, this would use value analysis
    return mlir::failure(); // Placeholder - requires more analysis
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Algebraic Optimizer Pass
//===----------------------------------------------------------------------===//

struct SoNAlgebraicOptPass
    : public mlir::PassWrapper<SoNAlgebraicOptPass, mlir::OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const override { return "son-algebraic-opt"; }
  StringRef getDescription() const override {
    return "Sea-of-Nodes algebraic optimization pass";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    mlir::RewritePatternSet patterns(context);

    // Register all algebraic rewrite patterns
    patterns.add<JoinAssociativityPattern>(context);
    patterns.add<JoinCommutativityPattern>(context);
    patterns.add<TransitionIdempotencePattern>(context);
    patterns.add<ChartInversePattern>(context);
    patterns.add<IndexedIdentityPattern>(context);
    patterns.add<DeadJoinElimPattern>(context);
    patterns.add<CoordinatesCanonicalPattern>(context);

    // Apply with greedy rewrite (iterates until fixed point)
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 10;  // Allow multiple passes for convergence
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace {
struct SoNAlgebraicOptPassRegistration {
  SoNAlgebraicOptPassRegistration() {
    PassRegistration<SoNAlgebraicOptPass>();
  }
};

static SoNAlgebraicOptPassRegistration registration;
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createSoNAlgebraicOptPass() {
  return std::make_unique<SoNAlgebraicOptPass>();
}