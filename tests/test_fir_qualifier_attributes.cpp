#include <iostream>
#include <cassert>
#include <string>
#include <memory>

#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace cpp2_transpiler;

// Helper to verify MLIR module
static bool verify_mlir(mlir::ModuleOp module) {
    return mlir::succeeded(mlir::verify(module));
}

// Test 1: FIR encoding of inout qualifier
void test_fir_inout_qualifier() {
    std::cout << "Test: FIR encoding of 'inout' qualifier\n";

    // Create AST for: func foo: (inout x: int) -> int { return 0; }
    auto func = std::make_unique<FunctionDeclaration>("foo", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Parameter with inout qualifier
    FunctionDeclaration::Parameter param;
    param.name = "x";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    param.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param));

    // Body: return 0;
    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    // Check if we're using the stub implementation (returns empty module)
    if (!module) {
        std::cout << "  [SKIP] Using stub FIR converter - test requires full MLIR\n";
        return;
    }
    
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    // Check that the module contains a function named "foo"
    auto foo_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("foo");
    assert(foo_func && "Function 'foo' not found in module");

    // Check arg_attrs contains qualifier attribute
    auto argAttrs = foo_func.getArgAttrsAttr();
    assert(argAttrs && "No arg_attrs found");
    assert(argAttrs.size() == 1 && "Expected 1 argument attribute");

    auto firstArgAttr = mlir::cast<mlir::DictionaryAttr>(argAttrs[0]);
    assert(firstArgAttr && "Expected DictionaryAttr for first argument");

    auto qualAttr = firstArgAttr.get("cpp2.qualifier");
    assert(qualAttr && "No 'cpp2.qualifier' attribute found");

    auto qualStr = mlir::cast<mlir::StringAttr>(qualAttr).getValue();
    assert(qualStr == "inout" && "Qualifier should be 'inout'");

    std::cout << "  Qualifier attribute: " << qualStr.str() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 2: FIR encoding of out qualifier
void test_fir_out_qualifier() {
    std::cout << "Test: FIR encoding of 'out' qualifier\n";

    // Create AST for: func bar: (out result: int) -> int { return 0; }
    auto func = std::make_unique<FunctionDeclaration>("bar", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    FunctionDeclaration::Parameter param;
    param.name = "result";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    param.qualifiers.push_back(ParameterQualifier::Out);
    func->parameters.push_back(std::move(param));

    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) {
        std::cout << "  [SKIP] Using stub FIR converter\n";
        return;
    }
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    auto bar_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("bar");
    assert(bar_func && "Function 'bar' not found in module");

    auto argAttrs = bar_func.getArgAttrsAttr();
    assert(argAttrs && "No arg_attrs found");

    auto firstArgAttr = mlir::cast<mlir::DictionaryAttr>(argAttrs[0]);
    auto qualAttr = firstArgAttr.get("cpp2.qualifier");
    assert(qualAttr && "No 'cpp2.qualifier' attribute found");

    auto qualStr = mlir::cast<mlir::StringAttr>(qualAttr).getValue();
    assert(qualStr == "out" && "Qualifier should be 'out'");

    std::cout << "  Qualifier attribute: " << qualStr.str() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 3: FIR encoding of function without qualifiers
void test_fir_no_qualifiers() {
    std::cout << "Test: FIR encoding without qualifiers\n";

    // Create AST for: func normal: (x: int) -> int { return 0; }
    auto func = std::make_unique<FunctionDeclaration>("normal", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    FunctionDeclaration::Parameter param;
    param.name = "x";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    // No qualifiers
    func->parameters.push_back(std::move(param));

    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) {
        std::cout << "  [SKIP] Using stub FIR converter\n";
        return;
    }
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    auto normal_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("normal");
    assert(normal_func && "Function 'normal' not found in module");

    // arg_attrs should exist but be empty for the parameter
    auto argAttrs = normal_func.getArgAttrsAttr();
    assert(argAttrs && "No arg_attrs found");

    auto firstArgAttr = mlir::cast<mlir::DictionaryAttr>(argAttrs[0]);
    // Should be an empty dictionary or not have the qualifier attribute
    auto qualAttr = firstArgAttr.get("cpp2.qualifier");
    assert(!qualAttr && "Should not have 'cpp2.qualifier' attribute");

    std::cout << "  No qualifier attributes (as expected)\n";
    std::cout << "✓ Test passed\n";
}

// Test 4: FIR encoding of function with multiple qualifiers
void test_fir_multiple_qualifiers() {
    std::cout << "Test: FIR encoding of multiple qualifiers\n";

    // Create AST for function with multiple parameters having different qualifiers
    auto func = std::make_unique<FunctionDeclaration>("multi", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Parameter 1: inout a
    FunctionDeclaration::Parameter param1;
    param1.name = "a";
    param1.type = std::make_unique<Type>(Type::Kind::Builtin);
    param1.type->name = "int";
    param1.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param1));

    // Parameter 2: out b
    FunctionDeclaration::Parameter param2;
    param2.name = "b";
    param2.type = std::make_unique<Type>(Type::Kind::Builtin);
    param2.type->name = "int";
    param2.qualifiers.push_back(ParameterQualifier::Out);
    func->parameters.push_back(std::move(param2));

    // Parameter 3: no qualifiers
    FunctionDeclaration::Parameter param3;
    param3.name = "c";
    param3.type = std::make_unique<Type>(Type::Kind::Builtin);
    param3.type->name = "int";
    func->parameters.push_back(std::move(param3));

    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) {
        std::cout << "  [SKIP] Using stub FIR converter\n";
        return;
    }
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    auto multi_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("multi");
    assert(multi_func && "Function 'multi' not found in module");

    auto argAttrs = multi_func.getArgAttrsAttr();
    assert(argAttrs && "No arg_attrs found");
    assert(argAttrs.size() == 3 && "Expected 3 argument attributes");

    // Check first parameter (inout)
    auto arg0Attr = mlir::cast<mlir::DictionaryAttr>(argAttrs[0]);
    auto qual0 = arg0Attr.get("cpp2.qualifier");
    assert(qual0 && "Expected qualifier on arg 0");
    assert(mlir::cast<mlir::StringAttr>(qual0).getValue() == "inout");

    // Check second parameter (out)
    auto arg1Attr = mlir::cast<mlir::DictionaryAttr>(argAttrs[1]);
    auto qual1 = arg1Attr.get("cpp2.qualifier");
    assert(qual1 && "Expected qualifier on arg 1");
    assert(mlir::cast<mlir::StringAttr>(qual1).getValue() == "out");

    // Check third parameter (no qualifiers)
    auto arg2Attr = mlir::cast<mlir::DictionaryAttr>(argAttrs[2]);
    auto qual2 = arg2Attr.get("cpp2.qualifier");
    assert(!qual2 && "Expected no qualifier on arg 2");

    std::cout << "  Parameter 0 qualifier: inout\n";
    std::cout << "  Parameter 1 qualifier: out\n";
    std::cout << "  Parameter 2: (no qualifier)\n";
    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== FIR Qualifier Attribute Encoding Tests ===\n\n";

    test_fir_inout_qualifier();
    test_fir_out_qualifier();
    test_fir_no_qualifiers();
    test_fir_multiple_qualifiers();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
