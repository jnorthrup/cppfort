#include "Cpp2Passes.h"
#include "Cpp2FIRDialect.h"
#include "Cpp2SONDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::cpp2fir;
using namespace mlir::sond;

namespace {

/// Convert FIR constant operations to SON constant operations
struct ConvertFIRConstantToSON : public OpConversionPattern<cpp2fir::ConstantOp> {
  using OpConversionPattern<cpp2fir::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cpp2fir::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<sond::ConstantOp>(
        op, op.getType(), op.getValue());
    return success();
  }
};

struct ConvertFIRAddToSON : public OpConversionPattern<cpp2fir::AddOp> {
  using OpConversionPattern<cpp2fir::AddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::AddOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::AddOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIRSubToSON : public OpConversionPattern<cpp2fir::SubOp> {
  using OpConversionPattern<cpp2fir::SubOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::SubOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::SubOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIRMulToSON : public OpConversionPattern<cpp2fir::MulOp> {
  using OpConversionPattern<cpp2fir::MulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::MulOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::MulOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIRDivToSON : public OpConversionPattern<cpp2fir::DivOp> {
  using OpConversionPattern<cpp2fir::DivOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::DivOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::DivOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIRAndToSON : public OpConversionPattern<cpp2fir::AndOp> {
  using OpConversionPattern<cpp2fir::AndOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::AndOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::AndOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIROrToSON : public OpConversionPattern<cpp2fir::OrOp> {
  using OpConversionPattern<cpp2fir::OrOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::OrOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::OrOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ConvertFIRNotToSON : public OpConversionPattern<cpp2fir::NotOp> {
  using OpConversionPattern<cpp2fir::NotOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::NotOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::NotOp>(op, op.getResult().getType(),
                                             adaptor.getInput());
    return success();
  }
};

struct ConvertFIRCmpToSON : public OpConversionPattern<cpp2fir::CmpOp> {
  using OpConversionPattern<cpp2fir::CmpOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::CmpOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::CmpOp>(op, op.getResult().getType(),
                                              adaptor.getLhs(), adaptor.getRhs(),
                                              op.getPredicateAttr());
    return success();
  }
};

struct ConvertFIRPhiToSON : public OpConversionPattern<cpp2fir::PhiOp> {
  using OpConversionPattern<cpp2fir::PhiOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(cpp2fir::PhiOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sond::PhiOp>(op, op.getResult().getType(), adaptor.getValues());
    return success();
  }
};

/// Convert FIR return operations to standard return operations
struct ConvertFIRReturnToStd : public OpConversionPattern<cpp2fir::ReturnOp> {
  using OpConversionPattern<cpp2fir::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cpp2fir::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Convert FIR function operations to standard function operations
struct ConvertFIRFuncToStd : public OpConversionPattern<cpp2fir::FuncOp> {
  using OpConversionPattern<cpp2fir::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cpp2fir::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // Create a new standard function with the same signature
    auto funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(),
        op.getSymName(),
        op.getFunctionType());

    // Inline the FIR function body into the new function
    rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pass to convert FIR dialect to SON dialect
struct ConvertFIRToSONPass
    : public PassWrapper<ConvertFIRToSONPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFIRToSONPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<sond::Cpp2SONDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<sond::Cpp2SONDialect>();
    target.addIllegalDialect<cpp2fir::Cpp2FIRDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertFIRConstantToSON>(&getContext());
    patterns.add<ConvertFIRAddToSON>(&getContext());
    patterns.add<ConvertFIRSubToSON>(&getContext());
    patterns.add<ConvertFIRMulToSON>(&getContext());
    patterns.add<ConvertFIRDivToSON>(&getContext());
    patterns.add<ConvertFIRAndToSON>(&getContext());
    patterns.add<ConvertFIROrToSON>(&getContext());
    patterns.add<ConvertFIRNotToSON>(&getContext());
    patterns.add<ConvertFIRCmpToSON>(&getContext());
    patterns.add<ConvertFIRPhiToSON>(&getContext());
    patterns.add<ConvertFIRReturnToStd>(&getContext());
    patterns.add<ConvertFIRFuncToStd>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "convert-fir-to-son"; }
  StringRef getDescription() const final {
    return "Convert FIR dialect operations to SON dialect operations";
  }
};

} // namespace

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createConvertFIRToSONPass() {
  return std::make_unique<ConvertFIRToSONPass>();
}

} // namespace cpp2
} // namespace mlir

// Global static registration - ensures pass is registered at startup
static mlir::PassRegistration<ConvertFIRToSONPass> pass;

void registerConvertFIRToSONPass() {
  // Registration happens via the static initializer above
  // This function ensures the translation unit is linked
}
