#pragma once

#include "ast.hpp"
#include "Cpp2FIRDialect.h"
#include "diagnostic_collector.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace cpp2_transpiler {

/// Converter from Cpp2 AST to MLIR FIR dialect
class ASTToFIRConverter {
public:
    explicit ASTToFIRConverter(mlir::MLIRContext* context,
                              DiagnosticCollector* diagnostics = nullptr);

    /// Convert a function declaration to a FIR module
    mlir::ModuleOp convertToFIR(const FunctionDeclaration& func);

    /// Convert a statement to FIR operations
    mlir::LogicalResult convertStatement(const Statement& stmt, mlir::OpBuilder& builder);

    /// Convert an expression to an MLIR value
    mlir::Value convertExpression(const Expression& expr, mlir::OpBuilder& builder);

    /// Convert a Cpp2 type to an MLIR type
    mlir::Type convertType(const Type& type);

    /// Get the diagnostic collector
    DiagnosticCollector* getDiagnostics() const { return diagnostics; }

private:
    mlir::MLIRContext* context;
    mlir::OpBuilder builder;
    DiagnosticCollector* diagnostics;  // Optional, may be null

    /// Helper to get source location string from expression
    std::string getLocationString(const Expression& expr) const;

    /// Helper to get source location string from statement
    std::string getLocationString(const Statement& stmt) const;
};

} // namespace cpp2_transpiler
