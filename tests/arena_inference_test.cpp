#include "../include/ast.hpp"
#include "../include/semantic_analyzer.hpp"
#include <iostream>
#include <cassert>
#include <memory>

using namespace cpp2_transpiler;

void test_simple_arena() {
    std::cout << "Running test_simple_arena...\n";
    // {
    //   v: std::vector = ...;
    // }
    // Should get arena ID.
    
    AST ast;
    
    auto var = std::make_unique<VariableDeclaration>("v", 2);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector";
    
    // Manually set EscapeInfo to NoEscape (simulating escape analysis pass)
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;
    
    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var), 2);
    
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(decl_stmt));
    
    auto func = std::make_unique<FunctionDeclaration>("main", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";
    func->body = std::move(block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* body = static_cast<BlockStatement*>(func_decl->body.get());
    
    if (body->arena_scope_id.has_value()) {
        std::cout << "  PASS: Arena ID assigned: " << *body->arena_scope_id << "\n";
        
        // Check variable
        auto* stmt = static_cast<DeclarationStatement*>(body->statements[0].get());
        auto* v = static_cast<VariableDeclaration*>(stmt->declaration.get());
        
        if (v->semantic_info && v->semantic_info->arena) {
             std::cout << "  PASS: Variable 'v' tagged with arena " << v->semantic_info->arena->scope_id << "\n";
             assert(v->semantic_info->arena->scope_id == *body->arena_scope_id);
        } else {
             std::cerr << "  FAIL: Variable 'v' NOT tagged\n";
             exit(1);
        }
    } else {
        std::cerr << "  FAIL: No arena assigned to block\n";
        exit(1);
    }
}

void test_nested_arena() {
    std::cout << "Running test_nested_arena...\n";
    // {
    //   outer: std::vector = ...; // NoEscape
    //   {
    //     inner: std::vector = ...; // NoEscape
    //   }
    // }
    // Should get two different arena IDs.
    
    AST ast;
    
    // Inner block
    auto inner_var = std::make_unique<VariableDeclaration>("inner", 4);
    inner_var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    inner_var->type->name = "std::vector";
    inner_var->escape_info = std::make_unique<EscapeInfo>();
    inner_var->escape_info->kind = EscapeKind::NoEscape;
    
    auto inner_stmt = std::make_unique<DeclarationStatement>(std::move(inner_var), 4);
    auto inner_block = std::make_unique<BlockStatement>(3);
    inner_block->statements.push_back(std::move(inner_stmt));
    
    // Outer block
    auto outer_var = std::make_unique<VariableDeclaration>("outer", 2);
    outer_var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    outer_var->type->name = "std::vector";
    outer_var->escape_info = std::make_unique<EscapeInfo>();
    outer_var->escape_info->kind = EscapeKind::NoEscape;
    
    auto outer_stmt = std::make_unique<DeclarationStatement>(std::move(outer_var), 2);
    auto outer_block = std::make_unique<BlockStatement>(1);
    outer_block->statements.push_back(std::move(outer_stmt));
    outer_block->statements.push_back(std::move(inner_block));
    
    auto func = std::make_unique<FunctionDeclaration>("nested", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";
    func->body = std::move(outer_block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* outer_body = static_cast<BlockStatement*>(func_decl->body.get());
    
    // Verify outer
    if (outer_body->arena_scope_id.has_value()) {
        std::cout << "  PASS: Outer block arena ID: " << *outer_body->arena_scope_id << "\n";
    } else {
        std::cerr << "  FAIL: Outer block has no arena\n";
        exit(1);
    }
    
    // Verify inner
    auto* inner_body = static_cast<BlockStatement*>(outer_body->statements[1].get());
    if (inner_body->arena_scope_id.has_value()) {
        std::cout << "  PASS: Inner block arena ID: " << *inner_body->arena_scope_id << "\n";
        if (*inner_body->arena_scope_id == *outer_body->arena_scope_id) {
             std::cerr << "  FAIL: Inner and Outer share same ID (expected distinct)\n";
             // They could share if optimized, but basic impl increments ID.
             exit(1);
        }
    } else {
        std::cerr << "  FAIL: Inner block has no arena\n";
        exit(1);
    }
}

int main() {
    try {
        test_simple_arena();
        test_nested_arena();
        std::cout << "All arena inference tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
