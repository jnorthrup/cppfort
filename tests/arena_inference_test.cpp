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

void test_escaping_value_no_arena() {
    std::cout << "Running test_escaping_value_no_arena...\n";
    // {
    //   v: std::vector = ...;  // EscapeToReturn - should NOT get arena
    // }
    
    AST ast;
    
    auto var = std::make_unique<VariableDeclaration>("result", 2);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::vector";
    
    // Set EscapeInfo to EscapeToReturn (escapes scope)
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::EscapeToReturn;
    
    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var), 2);
    
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(decl_stmt));
    
    auto func = std::make_unique<FunctionDeclaration>("get_data", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::UserDefined);
    func->return_type->name = "std::vector";
    func->body = std::move(block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* body = static_cast<BlockStatement*>(func_decl->body.get());
    auto* stmt = static_cast<DeclarationStatement*>(body->statements[0].get());
    auto* v = static_cast<VariableDeclaration*>(stmt->declaration.get());
    
    // Escaping value should NOT have arena
    if (v->semantic_info && v->semantic_info->arena.has_value()) {
        std::cerr << "  FAIL: Escaping variable should NOT have arena\n";
        exit(1);
    }
    std::cout << "  PASS: Escaping variable correctly excluded from arena\n";
}

void test_primitive_no_arena() {
    std::cout << "Running test_primitive_no_arena...\n";
    // {
    //   x: int = 42;  // Small primitive - should use stack, not arena
    // }
    
    AST ast;
    
    auto var = std::make_unique<VariableDeclaration>("x", 2);
    var->type = std::make_unique<Type>(Type::Kind::Builtin);
    var->type->name = "int";
    
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;
    
    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var), 2);
    
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(decl_stmt));
    
    auto func = std::make_unique<FunctionDeclaration>("primitives", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";
    func->body = std::move(block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* body = static_cast<BlockStatement*>(func_decl->body.get());
    auto* stmt = static_cast<DeclarationStatement*>(body->statements[0].get());
    auto* v = static_cast<VariableDeclaration*>(stmt->declaration.get());
    
    // Small primitives use stack, not arena
    if (v->semantic_info && v->semantic_info->arena.has_value()) {
        std::cerr << "  FAIL: Small primitive should NOT have arena (uses stack)\n";
        exit(1);
    }
    std::cout << "  PASS: Small primitive correctly uses stack (no arena)\n";
}

void test_string_arena() {
    std::cout << "Running test_string_arena...\n";
    // {
    //   s: std::string = "...";  // NoEscape string - should get arena
    // }
    
    AST ast;
    
    auto var = std::make_unique<VariableDeclaration>("s", 2);
    var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    var->type->name = "std::string";
    
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::NoEscape;
    
    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var), 2);
    
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(decl_stmt));
    
    auto func = std::make_unique<FunctionDeclaration>("string_test", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";
    func->body = std::move(block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* body = static_cast<BlockStatement*>(func_decl->body.get());
    
    if (body->arena_scope_id.has_value()) {
        std::cout << "  PASS: std::string gets arena ID: " << *body->arena_scope_id << "\n";
    } else {
        std::cerr << "  FAIL: std::string should get arena allocation\n";
        exit(1);
    }
}

void test_mixed_scope_allocation() {
    std::cout << "Running test_mixed_scope_allocation...\n";
    // {
    //   local_vec: std::vector = ...;   // NoEscape → arena
    //   result: std::vector = ...;      // EscapeToReturn → heap
    // }
    // Both in same block, but different allocation strategies
    
    AST ast;
    
    // NoEscape variable
    auto local_var = std::make_unique<VariableDeclaration>("local_vec", 2);
    local_var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    local_var->type->name = "std::vector";
    local_var->escape_info = std::make_unique<EscapeInfo>();
    local_var->escape_info->kind = EscapeKind::NoEscape;
    
    // Escaping variable  
    auto result_var = std::make_unique<VariableDeclaration>("result", 3);
    result_var->type = std::make_unique<Type>(Type::Kind::UserDefined);
    result_var->type->name = "std::vector";
    result_var->escape_info = std::make_unique<EscapeInfo>();
    result_var->escape_info->kind = EscapeKind::EscapeToReturn;
    
    auto local_stmt = std::make_unique<DeclarationStatement>(std::move(local_var), 2);
    auto result_stmt = std::make_unique<DeclarationStatement>(std::move(result_var), 3);
    
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(local_stmt));
    block->statements.push_back(std::move(result_stmt));
    
    auto func = std::make_unique<FunctionDeclaration>("mixed", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::UserDefined);
    func->return_type->name = "std::vector";
    func->body = std::move(block);
    
    ast.declarations.push_back(std::move(func));
    
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);
    
    auto* func_decl = static_cast<FunctionDeclaration*>(ast.declarations[0].get());
    auto* body = static_cast<BlockStatement*>(func_decl->body.get());
    
    auto* local_decl = static_cast<DeclarationStatement*>(body->statements[0].get());
    auto* local = static_cast<VariableDeclaration*>(local_decl->declaration.get());
    
    auto* result_decl = static_cast<DeclarationStatement*>(body->statements[1].get());
    auto* result = static_cast<VariableDeclaration*>(result_decl->declaration.get());
    
    bool local_has_arena = local->semantic_info && local->semantic_info->arena.has_value();
    bool result_has_arena = result->semantic_info && result->semantic_info->arena.has_value();
    
    if (local_has_arena && !result_has_arena) {
        std::cout << "  PASS: NoEscape → arena, EscapeToReturn → heap\n";
    } else {
        std::cerr << "  FAIL: Mixed allocation strategies not correct\n";
        std::cerr << "    local_has_arena=" << local_has_arena << ", result_has_arena=" << result_has_arena << "\n";
        exit(1);
    }
}

int main() {
    try {
        test_simple_arena();
        test_nested_arena();
        test_escaping_value_no_arena();
        test_primitive_no_arena();
        test_string_arena();
        test_mixed_scope_allocation();
        std::cout << "\n========================================\n";
        std::cout << "✅ All 6 arena inference tests passed!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
