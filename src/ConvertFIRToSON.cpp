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
