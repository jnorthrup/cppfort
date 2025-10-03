#pragma once

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <string>
#include <vector>
#include <string>
#include <vector>

namespace cppfort::mlir {

// Type aliases for better transparency and readability
using Context = ::mlir::MLIRContext;
using Module = ::mlir::ModuleOp;
using Function = ::mlir::func::FuncOp;
using Block = ::mlir::Block;
using Operation = ::mlir::Operation;
using Value = ::mlir::Value;
using Type = ::mlir::Type;
using Location = ::mlir::Location;

// Common type aliases
using IntegerType = ::mlir::IntegerType;
using FloatType = ::mlir::FloatType;
using IndexType = ::mlir::IndexType;
using MemRefType = ::mlir::MemRefType;
using TensorType = ::mlir::TensorType;
using VectorType = ::mlir::VectorType;

// Dialect operation aliases for direct abstraction
namespace arith {
using AddI = ::mlir::arith::AddIOp;
using SubI = ::mlir::arith::SubIOp;
using MulI = ::mlir::arith::MulIOp;
using DivSI = ::mlir::arith::DivSIOp;
using CmpI = ::mlir::arith::CmpIOp;
using Constant = ::mlir::arith::ConstantOp;
} // namespace arith

namespace func {
using Func = ::mlir::func::FuncOp;
using Call = ::mlir::func::CallOp;
using Return = ::mlir::func::ReturnOp;
} // namespace func

namespace cf {
using Branch = ::mlir::cf::BranchOp;
using CondBranch = ::mlir::cf::CondBranchOp;
} // namespace cf

namespace scf {
using If = ::mlir::scf::IfOp;
using For = ::mlir::scf::ForOp;
using While = ::mlir::scf::WhileOp;
} // namespace scf

namespace memref {
using Load = ::mlir::memref::LoadOp;
using Store = ::mlir::memref::StoreOp;
using Alloc = ::mlir::memref::AllocOp;
using Alloca = ::mlir::memref::AllocaOp;
// Direct abstraction layer for common operations
class MLIRBuilder {
private:
    ::mlir::OpBuilder builder_;
    Location loc_;

public:
    MLIRBuilder(::mlir::MLIRContext* context, Location loc = ::mlir::UnknownLoc())
        : builder_(context), loc_(loc) {}

    explicit MLIRBuilder(::mlir::OpBuilder builder, Location loc = ::mlir::UnknownLoc())
        : builder_(builder), loc_(loc) {}

    ::mlir::MLIRContext* getContext() const { return builder_.getContext(); }

    // Type creation helpers
    IntegerType getI32Type() { return builder_.getI32Type(); }
    IntegerType getI64Type() { return builder_.getI64Type(); }
    FloatType getF32Type() { return builder_.getF32Type(); }
    FloatType getF64Type() { return builder_.getF64Type(); }
    IndexType getIndexType() { return builder_.getIndexType(); }

    // Value creation helpers
    Value createConstant(int64_t value, Type type) {
        return builder_.create<::mlir::arith::ConstantOp>(loc_, type,
            builder_.getIntegerAttr(type, value)).getResult();
    }

    Value createConstant(int32_t value) {
        return createConstant(value, getI32Type());
    }

    Value createConstant(int64_t value) {
        return createConstant(value, getI64Type());
    }

    Value createConstant(float value) {
        auto type = getF32Type();
        return builder_.create<::mlir::arith::ConstantOp>(loc_, type,
            builder_.getFloatAttr(type, value)).getResult();
    }

    Value createConstant(double value) {
        auto type = getF64Type();
        return builder_.create<::mlir::arith::ConstantOp>(loc_, type,
            builder_.getFloatAttr(type, value)).getResult();
    }

    // Arithmetic operations
    Value createAdd(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::AddIOp>(loc_, lhs, rhs).getResult();
    }

    Value createSub(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::SubIOp>(loc_, lhs, rhs).getResult();
    }

    Value createMul(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::MulIOp>(loc_, lhs, rhs).getResult();
    }

    Value createDiv(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::DivSIOp>(loc_, lhs, rhs).getResult();
    }

    // Comparison operations
    Value createCmpEQ(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::eq, lhs, rhs).getResult();
    }

    Value createCmpNE(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::ne, lhs, rhs).getResult();
    }

    Value createCmpLT(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::slt, lhs, rhs).getResult();
    }

    Value createCmpLE(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::sle, lhs, rhs).getResult();
    }

    Value createCmpGT(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::sgt, lhs, rhs).getResult();
    }

    Value createCmpGE(Value lhs, Value rhs) {
        return builder_.create<::mlir::arith::CmpIOp>(loc_, ::mlir::arith::CmpIPredicate::sge, lhs, rhs).getResult();
    }

    // Function operations
    ::mlir::func::FuncOp createFunction(std::string name, ::mlir::FunctionType type) {
        return ::mlir::func::FuncOp::create(loc_, name, type);
    }

    void createReturn(Value value) {
        builder_.create<::mlir::func::ReturnOp>(loc_, value);
    }

    void createReturn() {
        builder_.create<::mlir::func::ReturnOp>(loc_);
    }

    // Control flow operations
    void createBranch(Block* dest, ::mlir::ArrayRef<Value> args = {}) {
        builder_.create<::mlir::cf::BranchOp>(loc_, dest, args);
    }

    void createCondBranch(Value condition, Block* dest,
                          ::mlir::ArrayRef<Value> args = {}) {
        builder_.create<::mlir::cf::CondBranchOp>(loc_, condition, dest, args);
    }

    // Structured control flow operations
    ::mlir::scf::IfOp createIf(Value condition, Type resultType = {}) {
        if (resultType) {
            return builder_.create<::mlir::scf::IfOp>(loc_, resultType, condition, /*withElse=*/false);
        } else {
            return builder_.create<::mlir::scf::IfOp>(loc_, condition, /*withElse=*/false);
        }
    }

    ::mlir::scf::ForOp createFor(Value lowerBound, Value upperBound, Value step,
                         ::mlir::function_ref<void(::mlir::OpBuilder&, Location, Value, ::mlir::ValueRange)> bodyBuilder) {
        return builder_.create<::mlir::scf::ForOp>(loc_, lowerBound, upperBound, step, std::nullopt, bodyBuilder);
    }

    // Memory operations
    Value createLoad(Value memref, Value index = {}) {
        if (index) {
            return builder_.create<::mlir::memref::LoadOp>(loc_, memref, index).getResult();
        } else {
            return builder_.create<::mlir::memref::LoadOp>(loc_, memref).getResult();
        }
    }

    void createStore(Value value, Value memref, Value index = {}) {
        if (index) {
            builder_.create<::mlir::memref::StoreOp>(loc_, value, memref, index);
        } else {
            builder_.create<::mlir::memref::StoreOp>(loc_, value, memref);
        }
    }

    Value createAlloc(MemRefType type, ::mlir::ValueRange dynamicSizes = {}) {
        return builder_.create<::mlir::memref::AllocOp>(loc_, type, dynamicSizes).getResult();
    }

    Value createAlloca(MemRefType type, ::mlir::ValueRange dynamicSizes = {}) {
        return builder_.create<::mlir::memref::AllocaOp>(loc_, type, dynamicSizes).getResult();
    }

    // Builder access for advanced operations
    ::mlir::OpBuilder& getBuilder() { return builder_; }
    const ::mlir::OpBuilder& getBuilder() const { return builder_; }
};

} // namespace cppfort::mlir