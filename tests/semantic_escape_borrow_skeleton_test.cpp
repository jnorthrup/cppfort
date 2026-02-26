#include "../include/ast.hpp"
#include "../include/semantic_analyzer.hpp"

#include <cassert>
#include <iostream>
#include <memory>

using namespace cpp2_transpiler;

static std::unique_ptr<Type> make_int_type() {
    auto t = std::make_unique<Type>(Type::Kind::Builtin);
    t->name = "int";
    return t;
}

int main() {
    AST ast;

    auto func = std::make_unique<FunctionDeclaration>("skeleton_walk", 10);
    func->return_type = make_int_type();

    auto block = std::make_unique<BlockStatement>(11);

    auto local = std::make_unique<VariableDeclaration>("x", 12);
    local->type = make_int_type();
    auto* local_ptr = local.get();

    block->statements.push_back(
        std::make_unique<DeclarationStatement>(std::move(local), 12));

    func->body = std::move(block);
    auto* func_ptr = func.get();
    ast.declarations.push_back(std::move(func));

    assert(func_ptr->semantic_info == nullptr);
    assert(local_ptr->semantic_info == nullptr);

    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);

    // Placeholder traversal should allocate declaration-level semantic info.
    assert(func_ptr->semantic_info != nullptr);
    assert(local_ptr->semantic_info != nullptr);

    // Placeholder pass must remain non-invasive/default-safe.
    assert(func_ptr->semantic_info->escape.kind == EscapeKind::NoEscape);
    assert(local_ptr->semantic_info->escape.kind == EscapeKind::NoEscape);

    std::cout << "semantic_escape_borrow_skeleton_test passed\n";
    return 0;
}
