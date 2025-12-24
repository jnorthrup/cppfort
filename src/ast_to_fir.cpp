#include "ast_to_fir.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace cpp2_transpiler;
using namespace mlir;
using namespace mlir::cpp2fir;

ASTToFIRConverter::ASTToFIRConverter(MLIRContext* ctx)
    : context(ctx), builder(ctx) {}

ModuleOp ASTToFIRConverter::convertToFIR(const FunctionDeclaration& func) {
    // Create a module
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Convert return type
    mlir::Type resultType = convertType(*func.return_type);

    // Convert parameter types
    SmallVector<mlir::Type> paramTypes;
    for (const auto& param : func.parameters) {
        paramTypes.push_back(convertType(*param.type));
    }

    // Create function type
    auto funcType = builder.getFunctionType(paramTypes, {resultType});

    // Create FIR function operation
    auto firFunc = builder.create<FuncOp>(
        loc,
        StringAttr::get(context, func.name),
        TypeAttr::get(funcType),
        ArrayAttr{},  // arg_attrs
        ArrayAttr{}   // res_attrs
    );

    // Create entry block
    Block* entry = firFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Convert body
    if (func.body) {
        if (failed(convertStatement(*func.body, builder))) {
            return nullptr;
        }
    }

    return module;
}

LogicalResult ASTToFIRConverter::convertStatement(const Statement& stmt, OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();

    if (auto* retStmt = dynamic_cast<const ReturnStatement*>(&stmt)) {
        if (retStmt->value) {
            Value val = convertExpression(*retStmt->value, builder);
            if (!val) {
                return failure();
            }
            builder.create<ReturnOp>(loc, ValueRange{val});
        } else {
            builder.create<ReturnOp>(loc, ValueRange{});
        }
        return success();
    }

    return failure();
}

Value ASTToFIRConverter::convertExpression(const Expression& expr, OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();

    if (auto* lit = dynamic_cast<const LiteralExpression*>(&expr)) {
        if (std::holds_alternative<int64_t>(lit->value)) {
            int64_t val = std::get<int64_t>(lit->value);
            auto i32Type = builder.getI32Type();
            auto attr = builder.getI32IntegerAttr(val);
            auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
            return constOp.getResult();
        } else if (std::holds_alternative<std::string>(lit->value)) {
            std::string val = std::get<std::string>(lit->value);
            // For now, create a constant with 0 (will be extended)
            auto i32Type = builder.getI32Type();
            auto attr = builder.getI32IntegerAttr(0);
            auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
            return constOp.getResult();
        }
    } else if (auto* ident = dynamic_cast<const IdentifierExpression*>(&expr)) {
        // Variable reference - for now return a dummy constant
        auto i32Type = builder.getI32Type();
        auto attr = builder.getI32IntegerAttr(0);
        auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
        return constOp.getResult();
    } else if (auto* binop = dynamic_cast<const BinaryExpression*>(&expr)) {
        Value lhs = convertExpression(*binop->left, builder);
        Value rhs = convertExpression(*binop->right, builder);

        if (!lhs || !rhs) {
            return nullptr;
        }

        // Create FIR add operation for Plus
        if (binop->op == TokenType::Plus) {
            // For now, return dummy constant since we don't have FIR add yet
            auto i32Type = builder.getI32Type();
            auto attr = builder.getI32IntegerAttr(0);
            auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
            return constOp.getResult();
        }

        // For other operators, return a constant as placeholder
        auto i32Type = builder.getI32Type();
        auto attr = builder.getI32IntegerAttr(0);
        auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
        return constOp.getResult();
    }

    return nullptr;
}

mlir::Type ASTToFIRConverter::convertType(const cpp2_transpiler::Type& type) {
    if (type.kind == cpp2_transpiler::Type::Kind::Builtin) {
        if (type.name == "int" || type.name == "i32") {
            return builder.getI32Type();
        }
        if (type.name == "int64" || type.name == "i64") {
            return builder.getI64Type();
        }
        if (type.name == "bool") {
            return builder.getI1Type();
        }
    }

    // Default to i32
    return builder.getI32Type();
}
