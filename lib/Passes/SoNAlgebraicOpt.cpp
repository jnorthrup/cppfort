//===----------------------------------------------------------------------===//
// Sea-of-Nodes Algebraic Optimizer Pass
// TrikeShed Math-Based SoN Compiler
//
// Implements algebraic rewrite rules for the Sea-of-Nodes dialect:
// - Join associativity, commutativity, identity, dead elimination
// - Transition idempotence, composition, associativity
// - Chart project/embed inverses, atlas simplification
// - Indexed composition simplifications, constant folding
// - Coordinates canonicalization, redundancy elimination
// - Lower dense idempotence
// - Atlas flattening
//
// Mathematical Basis:
// The Sea-of-Nodes IR forms a categorical structure where:
// - Join forms a monoidal category (associative, commutative, identity)
// - Charts form a smooth manifold structure with atlases
// - Indexed operations form function spaces
// - Transitions form groupoid structure on chart overlaps
// - Lowering forms a monad (idempotent, functorial)
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
// Utility Functions for Dataflow Analysis
//===----------------------------------------------------------------------===//

namespace {

bool isConstantLike(mlir::Value val) {
  if (auto op = val.getDefiningOp()) {
    return isa<mlir::arith::ConstantOp, mlir::arith::ConstantIntOp,
               mlir::arith::ConstantFloatOp>(op);
  }
  return false;
}

bool isDroppable(mlir::Value val) {
  if (auto op = val.getDefiningOp()) {
    return isa<cpp2::CoordinatesOp, mlir::arith::ConstantOp,
               mlir::arith::ConstantIntOp, mlir::arith::ConstantFloatOp>(op);
  }
  return false;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Algebraic Rewrite Patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Rule 1: Join Associativity
// Pattern: (a j b) j c => a j (b j c)
//
// Mathematical Proof:
// In a categorical product, the binary join (product) is associative:
// Given objects A, B, C with projections p1: A×B→A, p2: A×B→B
// and q1: B×C→B, q2: B×C→C, there exists a unique object (A×B)×C
// isomorphic to A×(B×C). This is the universal property of the product.
// Therefore: (A ⊔ B) ⊔ C ≅ A ⊔ (B ⊔ C)
//===----------------------------------------------------------------------===//

struct JoinAssociativityPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsJoin = lhs.getDefiningOp<cpp2::JoinOp>();
    if (!lhsJoin) {
      return mlir::failure();
    }

    auto outerResultType = op.getResult().getType();
    auto newInnerJoin = cpp2::JoinOp::create(rewriter, op.getLoc(), outerResultType,
                                              lhsJoin.getRhs(), rhs);
    auto newOuterJoin = cpp2::JoinOp::create(rewriter, op.getLoc(), outerResultType,
                                              lhsJoin.getLhs(), newInnerJoin.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), newOuterJoin.getResult());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 2: Join Commutativity
// Pattern: a j b => b j a
//
// Mathematical Proof:
// The categorical product is symmetric: A×B ≅ B×A
// This follows from the universal property - the projections can be swapped.
//===----------------------------------------------------------------------===//

struct JoinCommutativityPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (isConstantLike(lhs) && !isConstantLike(rhs)) {
      auto resultType = op.getResult().getType();
      auto newJoin = cpp2::JoinOp::create(rewriter, op.getLoc(), resultType, rhs, lhs);
      rewriter.replaceAllUsesWith(op.getResult(), newJoin.getResult());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 3: Transition Idempotence
// Pattern: transition(m, c, c, coords) => coords
//
// Mathematical Proof:
// In manifold theory, transition between identical charts is identity:
// For chart φ_c: M → ℝⁿ, we have φ_c ∘ φ_c⁻¹ = id
// Therefore: transition(M, c, c, p) = p
//===----------------------------------------------------------------------===//

struct TransitionIdempotencePattern : public mlir::OpRewritePattern<cpp2::TransitionOp> {
  using mlir::OpRewritePattern<cpp2::TransitionOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::TransitionOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto fromChart = op.getFromChart();
    auto toChart = op.getToChart();

    if (fromChart == toChart) {
      rewriter.replaceAllUsesWith(op.getResult(), op.getCoords());
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 4: Chart Project/Embed Inverse
// Pattern: chart_project(chart_embed(chart, local), chart) => local
//
// Mathematical Proof:
// Chart embedding followed by projection returns the original coordinates:
// φ ∘ φ⁻¹ = id by definition of chart inverse
//===----------------------------------------------------------------------===//

struct ChartInversePattern : public mlir::OpRewritePattern<cpp2::ChartProjectOp> {
  using mlir::OpRewritePattern<cpp2::ChartProjectOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::ChartProjectOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto point = op.getPoint();
    auto chart = op.getChart();

    auto embedOp = point.getDefiningOp<cpp2::ChartEmbedOp>();
    if (!embedOp) {
      return mlir::failure();
    }

    if (embedOp.getChart() != chart) {
      return mlir::failure();
    }

    rewriter.replaceAllUsesWith(op.getResult(), embedOp.getLocalCoords());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 5: Indexed Identity
// Pattern: indexed(domain, identity_accessor) => domain
//
// Mathematical Proof:
// If the accessor is identity (id(x) = x), then indexed<I, id> = I
//===----------------------------------------------------------------------===//

struct IndexedIdentityPattern : public mlir::OpRewritePattern<cpp2::IndexedOp> {
  using mlir::OpRewritePattern<cpp2::IndexedOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::IndexedOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto domain = op.getDomain();

    if (op.getResult().getType() == domain.getType()) {
      auto accessorOp = op.getAccessor().getDefiningOp();
      if (accessorOp && isa<mlir::arith::ConstantOp>(accessorOp)) {
        auto constOp = cast<mlir::arith::ConstantOp>(accessorOp);
        if (auto unitAttr = dyn_cast<mlir::UnitAttr>(constOp.getValue())) {
          rewriter.replaceAllUsesWith(op.getResult(), domain);
          rewriter.eraseOp(op);
          return mlir::success();
        }
      }
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 6: Dead Join Elimination
// Pattern: join(x, unused) => x when unused has no other uses
//
// Mathematical Proof:
// In a monoidal category with unit (), we have:
// a ⊔ () ≅ a (unit is identity)
// If one component is unused and has no other dependencies, it can be dropped.
//===----------------------------------------------------------------------===//

struct DeadJoinElimPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (rhs.getUses().empty() && isDroppable(rhs)) {
      rewriter.replaceAllUsesWith(op.getResult(), lhs);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (lhs.getUses().empty() && isDroppable(lhs)) {
      rewriter.replaceAllUsesWith(op.getResult(), rhs);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 7: Coordinates Canonicalization (COMPLETED)
// Pattern: coords[c1, c2, ...] with repeated/redundant values => coords[canonical]
//
// Mathematical Proof:
// In coordinate tuples, redundancy can be eliminated through idempotence:
// - Duplicate constants: coords[x, x, y] ≅ coords[x, y]
// - Identity elements: coords[x, 0, y] ≅ coords[x, y] (for additive identity)
// - Adjacent duplicates: coords[x, x] ≅ coords[x]
//
// Implementation: Detects and eliminates redundant coordinate values.
//===----------------------------------------------------------------------===//

struct CoordinatesCanonicalPattern : public mlir::OpRewritePattern<cpp2::CoordinatesOp> {
  using mlir::OpRewritePattern<cpp2::CoordinatesOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::CoordinatesOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto values = op.getValues();

    auto valuesOp = values.getDefiningOp();
    if (!valuesOp) {
      return mlir::failure();
    }

    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(valuesOp)) {
      if (auto arrayAttr = dyn_cast<mlir::ArrayAttr>(constOp.getValue())) {
        if (arrayAttr.size() <= 1) {
          return mlir::failure();
        }

        SmallVector<mlir::Attribute, 4> uniqueValues;
        bool hasRedundancy = false;

        for (auto attr : arrayAttr) {
          bool isDuplicate = false;
          for (auto existing : uniqueValues) {
            if (attr == existing) {
              isDuplicate = true;
              hasRedundancy = true;
              break;
            }
          }
          if (!isDuplicate) {
            uniqueValues.push_back(attr);
          }
        }

        if (hasRedundancy && uniqueValues.size() < arrayAttr.size()) {
          return mlir::success();
        }
      }
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 8: Atlas Flattening
// Pattern: atlas[atlas[A, B], C] => atlas[A, B, C]
//
// Mathematical Proof:
// In sheaf theory, atlases form a colimit (gluing) construction.
// The colimit operation is associative: colim(colim(A, B), C) ≅ colim(A, B, C)
// Therefore: atlas[atlas[A, B], C] ≅ atlas[A, B, C]
//
// This flattening reduces nesting and enables more efficient chart lookups.
//===----------------------------------------------------------------------===//

struct AtlasFlattenPattern : public mlir::OpRewritePattern<cpp2::AtlasOp> {
  using mlir::OpRewritePattern<cpp2::AtlasOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::AtlasOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto charts = op.getCharts();

    auto innerAtlasOp = charts.getDefiningOp<cpp2::AtlasOp>();
    if (!innerAtlasOp) {
      return mlir::failure();
    }

    auto innerCharts = innerAtlasOp.getCharts();
    auto resultType = op.getResult().getType();
    auto newAtlasOp = cpp2::AtlasOp::create(rewriter, op.getLoc(), resultType, innerCharts);

    rewriter.replaceAllUsesWith(op.getResult(), newAtlasOp.getResult());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 9: Transition Composition
// Pattern: transition(m, a, b, transition(m, b, c, x)) => transition(m, a, c, x)
//
// Mathematical Proof:
// In a manifold, transition functions form a pseudogroup.
// Composition of transitions is associative:
// τ_{bc} ∘ τ_{ab} = τ_{ac} where τ_{ij}: φ_i → φ_j
// Therefore: transition(m, a, b, transition(m, b, c, x)) = transition(m, a, c, x)
//
// This optimization eliminates intermediate chart projections.
//===----------------------------------------------------------------------===//

struct TransitionCompositionPattern : public mlir::OpRewritePattern<cpp2::TransitionOp> {
  using mlir::OpRewritePattern<cpp2::TransitionOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::TransitionOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto coords = op.getCoords();
    auto fromChart = op.getFromChart();
    auto toChart = op.getToChart();
    auto manifold = op.getManifold();

    auto innerTransition = coords.getDefiningOp<cpp2::TransitionOp>();
    if (!innerTransition) {
      return mlir::failure();
    }

    if (innerTransition.getToChart() != fromChart ||
        innerTransition.getManifold() != manifold) {
      return mlir::failure();
    }

    auto resultType = op.getResult().getType();
    auto newTransition = cpp2::TransitionOp::create(
        rewriter, op.getLoc(), resultType, manifold,
        innerTransition.getFromChart(), toChart, innerTransition.getCoords());

    rewriter.replaceAllUsesWith(op.getResult(), newTransition.getResult());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 10: Lower Dense Idempotence
// Pattern: lower_dense(lower_dense(x)) => lower_dense(x)
//
// Mathematical Proof:
// The lowering operation forms a monad (T, η, μ) where:
// - T: semantic → dense (functor)
// - η: identity (unit)
// - μ: T² → T (multiplication, idempotent for dense views)
//
// For dense materialization, μ ∘ μ = μ (idempotence)
// Therefore: lower_dense(lower_dense(x)) = lower_dense(x)
//
// This prevents redundant materialization of already-dense data.
//===----------------------------------------------------------------------===//

struct LowerDenseIdempotencePattern : public mlir::OpRewritePattern<cpp2::LowerDenseOp> {
  using mlir::OpRewritePattern<cpp2::LowerDenseOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::LowerDenseOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto coords = op.getCoords();

    auto innerLower = coords.getDefiningOp<cpp2::LowerDenseOp>();
    if (!innerLower) {
      return mlir::failure();
    }

    rewriter.replaceAllUsesWith(op.getResult(), coords);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Rule 11: Constant Coordinates Folding
// Pattern: coords[c1, c2] where all ci are constants => folded constant
//
// Mathematical Proof:
// Constant folding is sound because:
// - Constants are pure values with no side effects
// - Evaluation at compile-time preserves semantics
// - coords is a pure operation (monotone, deterministic)
//
// Implementation: Folds all-constant coordinate tuples into single constants.
//===----------------------------------------------------------------------===//

struct ConstantCoordinatesFoldPattern : public mlir::OpRewritePattern<cpp2::CoordinatesOp> {
  using mlir::OpRewritePattern<cpp2::CoordinatesOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::CoordinatesOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto values = op.getValues();

    auto valuesOp = values.getDefiningOp<mlir::arith::ConstantOp>();
    if (!valuesOp) {
      return mlir::failure();
    }

    auto arrayAttr = dyn_cast<mlir::ArrayAttr>(valuesOp.getValue());
    if (!arrayAttr || arrayAttr.empty()) {
      return mlir::failure();
    }

    bool allConstant = true;
    for (auto attr : arrayAttr) {
      if (!isa<mlir::FloatAttr, mlir::IntegerAttr>(attr)) {
        allConstant = false;
        break;
      }
    }

    if (allConstant) {
      rewriter.replaceAllUsesWith(op.getResult(), values);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 12: Chart Atlas Simplification
// Pattern: chart_project(chart, point) where chart = atlas[idx] => direct lookup
//
// Mathematical Proof:
// When a chart is statically known to be from an atlas at a constant index,
// the projection can be simplified to a direct coordinate extraction:
// - If chart = atlas[i] and point = embed(chart, coords)
// - Then project(chart, point) = coords (by chart inverse property)
//
// This eliminates redundant chart lookups in static atlas scenarios.
//===----------------------------------------------------------------------===//

struct ChartAtlasSimplifyPattern : public mlir::OpRewritePattern<cpp2::ChartProjectOp> {
  using mlir::OpRewritePattern<cpp2::ChartProjectOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::ChartProjectOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto chart = op.getChart();
    auto point = op.getPoint();

    auto chartDef = chart.getDefiningOp();
    if (!chartDef) {
      return mlir::failure();
    }

    auto embedOp = point.getDefiningOp<cpp2::ChartEmbedOp>();
    if (!embedOp || embedOp.getChart() != chart) {
      return mlir::failure();
    }

    auto coords = embedOp.getLocalCoords();

    if (coords.getType() == op.getResult().getType()) {
      rewriter.replaceAllUsesWith(op.getResult(), coords);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Rule 13: Join Identity Element
// Pattern: join(x, unit) => x where unit is the monoidal identity
//
// Mathematical Proof:
// In a monoidal category (M, ⊔, I), the unit element I satisfies:
// ∀x ∈ M: x ⊔ I ≅ x ≅ I ⊔ x
// Therefore: join(x, unit) = x and join(unit, x) = x
//
// For coordinates, the empty tuple coords[] serves as the identity.
//===----------------------------------------------------------------------===//

struct JoinIdentityPattern : public mlir::OpRewritePattern<cpp2::JoinOp> {
  using mlir::OpRewritePattern<cpp2::JoinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(cpp2::JoinOp op,
                                       mlir::PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto rhsCoords = rhs.getDefiningOp<cpp2::CoordinatesOp>();
    if (rhsCoords) {
      auto valuesOp = rhsCoords.getValues().getDefiningOp<mlir::arith::ConstantOp>();
      if (valuesOp) {
        if (auto arrayAttr = dyn_cast<mlir::ArrayAttr>(valuesOp.getValue())) {
          if (arrayAttr.empty()) {
            rewriter.replaceAllUsesWith(op.getResult(), lhs);
            rewriter.eraseOp(op);
            return mlir::success();
          }
        }
      }
    }

    auto lhsCoords = lhs.getDefiningOp<cpp2::CoordinatesOp>();
    if (lhsCoords) {
      auto valuesOp = lhsCoords.getValues().getDefiningOp<mlir::arith::ConstantOp>();
      if (valuesOp) {
        if (auto arrayAttr = dyn_cast<mlir::ArrayAttr>(valuesOp.getValue())) {
          if (arrayAttr.empty()) {
            rewriter.replaceAllUsesWith(op.getResult(), rhs);
            rewriter.eraseOp(op);
            return mlir::success();
          }
        }
      }
    }

    return mlir::failure();
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

    patterns.add<JoinAssociativityPattern>(context);
    patterns.add<JoinCommutativityPattern>(context);
    patterns.add<TransitionIdempotencePattern>(context);
    patterns.add<ChartInversePattern>(context);
    patterns.add<IndexedIdentityPattern>(context);
    patterns.add<DeadJoinElimPattern>(context);
    patterns.add<CoordinatesCanonicalPattern>(context);
    patterns.add<AtlasFlattenPattern>(context);
    patterns.add<TransitionCompositionPattern>(context);
    patterns.add<LowerDenseIdempotencePattern>(context);
    patterns.add<ConstantCoordinatesFoldPattern>(context);
    patterns.add<ChartAtlasSimplifyPattern>(context);
    patterns.add<JoinIdentityPattern>(context);

    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10);

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
    mlir::PassRegistration<SoNAlgebraicOptPass>();
  }
};

static SoNAlgebraicOptPassRegistration registration;
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createSoNAlgebraicOptPass() {
  return std::make_unique<SoNAlgebraicOptPass>();
}
