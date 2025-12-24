#pragma once

#include "ast.hpp"
#include "Cpp2FIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace cpp2_transpiler {

/// Converter from Cpp2 AST to MLIR FIR dialect
class ASTToFIRConverter {
public:
    explicit ASTToFIRConverter(mlir::MLIRContext* context);

    /// Convert a function declaration to a FIR module
    mlir::ModuleOp convertToFIR(const FunctionDeclaration& func);

    /// Convert a statement to FIR operations
    mlir::LogicalResult convertStatement(const Statement& stmt, mlir::OpBuilder& builder);

    /// Convert an expression to an MLIR value
    mlir::Value convertExpression(const Expression& expr, mlir::OpBuilder& builder);

    /// Convert a Cpp2 type to an MLIR type
    mlir::Type convertType(const Type& type);

private:
    mlir::MLIRContext* context;
    mlir::OpBuilder builder;
};

} // namespace cpp2_transpiler
