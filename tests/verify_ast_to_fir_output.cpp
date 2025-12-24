#include <iostream>
#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace cpp2_transpiler;

int main() {
    // Create AST for: main: () -> int = 42;
    auto func = std::make_unique<FunctionDeclaration>("main", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(42), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    // Print the module
    module.print(llvm::outs());
    std::cout << "\n";

    return 0;
}
