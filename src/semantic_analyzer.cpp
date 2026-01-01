#include "semantic_analyzer.hpp"
#include <iostream>
#include <format>
#include <algorithm>

namespace cpp2_transpiler {

SemanticAnalyzer::SemanticAnalyzer() {
    register_builtin_types();
}

void SemanticAnalyzer::analyze(AST& ast) {
    current_ast = &ast;
    has_cpp1_passthrough = false;
    push_scope(); // Global scope

    // First pass: register all declarations
    for (auto& decl : ast.declarations) {
        if (decl) {
            // Check if this is a C++1 passthrough declaration
            if (decl->kind == Declaration::Kind::Cpp1Passthrough) {
                has_cpp1_passthrough = true;
            }
            if (auto ns = dynamic_cast<NamespaceDeclaration*>(decl.get())) {
                add_symbol(ns->name, std::make_unique<Symbol>(
                    Symbol::Kind::Namespace, ns->name, nullptr, ns));
            } else if (auto type_decl = dynamic_cast<TypeDeclaration*>(decl.get())) {
                auto type = std::make_unique<Type>(Type::Kind::UserDefined);
                type->name = type_decl->name;
                auto symbol = std::make_unique<Symbol>(
                    Symbol::Kind::Type, type_decl->name, type.get(), type_decl);
                add_symbol(type_decl->name, std::move(symbol));
                builtin_types[type_decl->name] = std::move(type);
            } else if (auto func = dynamic_cast<FunctionDeclaration*>(decl.get())) {
                // Register function with type to be determined later
                auto func_type = std::make_unique<Type>(Type::Kind::Function);
                auto symbol = std::make_unique<Symbol>(
                    Symbol::Kind::Function, func->name, func_type.get(), func);
                add_symbol(func->name, std::move(symbol));
            }
        }
    }

    // Second pass: check all declarations
    for (auto& decl : ast.declarations) {
        if (decl && !checked_declarations[decl.get()]) {
            check_declaration(decl.get());
            checked_declarations[decl.get()] = true;
        }
    }

    // Check for unused variables
    check_unused_variables();

    if (!errors.empty()) {
        std::cerr << "Semantic analysis found " << errors.size() << " errors:\n";
        for (const auto& error : errors) {
            std::cerr << "  " << error << "\n";
        }
    }

    if (!warnings.empty()) {
        std::cerr << "Semantic analysis found " << warnings.size() << " warnings:\n";
        for (const auto& warning : warnings) {
            std::cerr << "  " << warning << "\n";
        }
    }

    if (!errors.empty()) {
        throw std::runtime_error("Semantic analysis failed");
    }
}

void SemanticAnalyzer::check_declaration(Declaration* decl) {
    if (!decl) return;

    switch (decl->kind) {
        case Declaration::Kind::Variable:
            check_variable_usage(static_cast<VariableDeclaration*>(decl));
            break;
        case Declaration::Kind::Function:
            check_function(static_cast<FunctionDeclaration*>(decl));
            break;
        case Declaration::Kind::Type:
            resolve_type_declaration(static_cast<TypeDeclaration*>(decl));
            break;
        case Declaration::Kind::Namespace: {
            push_scope();
            auto ns = static_cast<NamespaceDeclaration*>(decl);
            for (auto& member : ns->members) {
                check_declaration(member.get());
            }
            pop_scope();
            break;
        }
        case Declaration::Kind::Cpp1Passthrough:
            // Skip C++1 passthrough declarations - they're emitted as-is
            // and their types/symbols are not visible to Cpp2 code
            break;
        default:
            // Handle other declaration types
            break;
    }
}

void SemanticAnalyzer::check_expression(Expression* expr) {
    if (!expr) return;

    switch (expr->kind) {
        case Expression::Kind::Literal:
            check_literal_expression(static_cast<LiteralExpression*>(expr));
            break;
        case Expression::Kind::Identifier:
            check_identifier_expression(static_cast<IdentifierExpression*>(expr));
            break;
        case Expression::Kind::Binary:
            check_binary_expression(static_cast<BinaryExpression*>(expr));
            break;
        case Expression::Kind::Unary:
            check_unary_expression(static_cast<UnaryExpression*>(expr));
            break;
        case Expression::Kind::Call:
            check_call_expression(static_cast<CallExpression*>(expr));
            break;
        case Expression::Kind::MemberAccess:
            check_member_access_expression(static_cast<MemberAccessExpression*>(expr));
            break;
        case Expression::Kind::Subscript:
            check_subscript_expression(static_cast<SubscriptExpression*>(expr));
            break;
        case Expression::Kind::Ternary:
            check_ternary_expression(static_cast<TernaryExpression*>(expr));
            break;
        case Expression::Kind::Lambda:
            check_lambda_expression(static_cast<LambdaExpression*>(expr));
            break;
        case Expression::Kind::Is:
            check_is_expression(static_cast<IsExpression*>(expr));
            break;
        case Expression::Kind::As:
            check_as_expression(static_cast<AsExpression*>(expr));
            break;
        case Expression::Kind::Range:
            check_range_expression(static_cast<RangeExpression*>(expr));
            break;
        // Concurrency expressions
        case Expression::Kind::Await:
            check_await_expression(static_cast<AwaitExpression*>(expr));
            break;
        case Expression::Kind::Spawn:
            check_spawn_expression(static_cast<SpawnExpression*>(expr));
            break;
        case Expression::Kind::ChannelSend:
            check_channel_send_expression(static_cast<ChannelSendExpression*>(expr));
            break;
        case Expression::Kind::ChannelRecv:
            check_channel_recv_expression(static_cast<ChannelRecvExpression*>(expr));
            break;
        case Expression::Kind::ChannelSelect:
            check_channel_select_expression(static_cast<ChannelSelectExpression*>(expr));
            break;
        default:
            // Handle other expression types
            break;
    }
}

void SemanticAnalyzer::check_statement(Statement* stmt) {
    if (!stmt) return;

    switch (stmt->kind) {
        case Statement::Kind::Expression:
            check_expression(static_cast<ExpressionStatement*>(stmt)->expr.get());
            break;
        case Statement::Kind::Declaration: {
            auto decl_stmt = static_cast<DeclarationStatement*>(stmt);
            check_declaration(decl_stmt->declaration.get());
            break;
        }
        case Statement::Kind::Block: {
            push_scope();
            auto block = static_cast<BlockStatement*>(stmt);
            for (auto& s : block->statements) {
                check_statement(s.get());
            }
            pop_scope();
            break;
        }
        case Statement::Kind::If: {
            auto if_stmt = static_cast<IfStatement*>(stmt);
            check_expression(if_stmt->condition.get());
            check_statement(if_stmt->then_stmt.get());
            check_statement(if_stmt->else_stmt.get());
            break;
        }
        case Statement::Kind::While: {
            auto while_stmt = static_cast<WhileStatement*>(stmt);
            check_expression(while_stmt->condition.get());
            check_statement(while_stmt->body.get());
            break;
        }
        case Statement::Kind::For: {
            auto for_stmt = static_cast<ForStatement*>(stmt);
            if (for_stmt->init) check_statement(for_stmt->init.get());
            if (for_stmt->condition) check_expression(for_stmt->condition.get());
            if (for_stmt->increment) check_expression(for_stmt->increment.get());
            check_statement(for_stmt->body.get());
            break;
        }
        case Statement::Kind::ForRange: {
            auto range_stmt = static_cast<ForRangeStatement*>(stmt);
            // Range-for introduces a new variable scope for the loop variable.
            push_scope();

            if (range_stmt->var_type) {
                range_stmt->var_type = check_type(std::move(range_stmt->var_type));
            }

            // Add loop variable to scope so it can be referenced inside the loop body.
            {
                auto symbol = std::make_unique<Symbol>(
                    Symbol::Kind::Variable,
                    range_stmt->variable,
                    range_stmt->var_type.get(),
                    nullptr);
                add_symbol(range_stmt->variable, std::move(symbol));
                variable_usage[range_stmt->variable] = VariableUsage{};
            }

            check_expression(range_stmt->range.get());
            check_statement(range_stmt->body.get());

            pop_scope();
            break;
        }
        case Statement::Kind::Return: {
            auto return_stmt = static_cast<ReturnStatement*>(stmt);
            check_expression(return_stmt->value.get());
            break;
        }
        case Statement::Kind::Try: {
            auto try_stmt = static_cast<TryStatement*>(stmt);
            check_statement(try_stmt->try_block.get());
            for (auto& catch_block : try_stmt->catch_blocks) {
                check_statement(catch_block.second.get());
            }
            break;
        }
        case Statement::Kind::Contract: {
            auto contract_stmt = static_cast<ContractStatement*>(stmt);
            check_contract(contract_stmt->contract.get());
            break;
        }
        // Concurrency statements
        case Statement::Kind::CoroutineScope: {
            auto scope_stmt = static_cast<CoroutineScopeStatement*>(stmt);
            // Enter a new scope for the coroutine scope
            push_scope();
            in_coroutine_scope_depth++;
            check_statement(scope_stmt->body.get());
            in_coroutine_scope_depth--;
            pop_scope();
            break;
        }
        case Statement::Kind::ParallelFor: {
            auto par_stmt = static_cast<ParallelForStatement*>(stmt);
            check_expression(par_stmt->lower_bound.get());
            check_expression(par_stmt->upper_bound.get());
            if (par_stmt->step) check_expression(par_stmt->step.get());
            
            // Create new scope for loop variable
            push_scope();
            auto loop_var_type = std::make_unique<Type>(Type::Kind::Auto);
            auto symbol = std::make_unique<Symbol>(
                Symbol::Kind::Variable, par_stmt->loop_variable, loop_var_type.get(), nullptr);
            add_symbol(par_stmt->loop_variable, std::move(symbol));
            
            check_statement(par_stmt->body.get());
            pop_scope();
            break;
        }
        case Statement::Kind::ChannelDecl: {
            auto ch_stmt = static_cast<ChannelDeclarationStatement*>(stmt);
            // Register the channel in the current scope
            auto channel_type = std::make_unique<Type>(Type::Kind::UserDefined);
            channel_type->name = "cpp2::Channel";
            auto symbol = std::make_unique<Symbol>(
                Symbol::Kind::Variable, ch_stmt->name, channel_type.get(), nullptr);
            add_symbol(ch_stmt->name, std::move(symbol));
            break;
        }
        default:
            // Handle other statement types
            break;
    }
}

void SemanticAnalyzer::check_function(FunctionDeclaration* func) {
    if (!func) return;

    push_scope();

    // Add template parameters as types in this scope
    for (const std::string& tparam : func->template_parameters) {
        auto template_type = std::make_unique<Type>(Type::Kind::UserDefined);
        template_type->name = tparam;
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Type, tparam, template_type.get(), nullptr);
        add_symbol(tparam, std::move(symbol));
    }

    if (func->return_type) {
        func->return_type = check_type(std::move(func->return_type));
    }

    // Cpp2 contracts may reference a special `result` name in postconditions.
    // Provide it in the function scope so contract expressions type-check.
    if (func->return_type && func->return_type->name != "void") {
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Variable, "result", func->return_type.get(), nullptr);
        add_symbol("result", std::move(symbol));
    }

    // Add named return parameters to scope (Cpp2: -> (name: type))
    // These are variables that can be assigned in the function body
    for (auto& named_ret : func->named_returns) {
        if (named_ret.type) {
            named_ret.type = check_type(std::move(named_ret.type));
        }
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Variable, named_ret.name, named_ret.type.get(), nullptr);
        add_symbol(named_ret.name, std::move(symbol));
        // Track usage - named returns are implicitly used by the return
        variable_usage[named_ret.name] = VariableUsage{true, false, 0};
    }

    // Add parameters to scope
    for (auto& param : func->parameters) {
        if (param.type) {
            param.type = check_type(std::move(param.type));
        }
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Parameter, param.name, param.type.get(), nullptr);
        add_symbol(param.name, std::move(symbol));
    }

    check_function_contracts(func);
    check_parameter_types(func);

    if (func->body) {
        check_function_body(func);
    }

    pop_scope();
}

void SemanticAnalyzer::check_function_body(FunctionDeclaration* func) {
    check_statement(func->body.get());
}

void SemanticAnalyzer::check_parameter_types(FunctionDeclaration* func) {
    // Cpp2 allows parameters without explicit types - they become deduced/template parameters
    // So we don't require type annotations for parameters
    // for (const auto& param : func->parameters) {
    //     if (!param.type) {
    //         report_error(func->line, "Parameter '" + param.name + "' missing type annotation");
    //     }
    // }
}

void SemanticAnalyzer::check_function_contracts(FunctionDeclaration* func) {
    // Contract checking would be implemented here
}

void SemanticAnalyzer::check_variable_usage(VariableDeclaration* var) {
    if (!var) return;

    if (!var->type && !var->initializer) {
        report_error(var->line, "Variable '" + var->name + "' needs type or initializer");
    }

    if (var->initializer) {
        check_expression(var->initializer.get());
    }

    if (var->type) {
        var->type = check_type(std::move(var->type));
    }

    // Add variable to current scope
    auto symbol = std::make_unique<Symbol>(
        Symbol::Kind::Variable, var->name, var->type.get(), var);
    symbol->is_const = var->is_const;
    symbol->is_mut = var->is_mut;
    add_symbol(var->name, std::move(symbol));

    // Track usage
    variable_usage[var->name] = VariableUsage{};
}

void SemanticAnalyzer::check_literal_expression(LiteralExpression* expr) {
    // Type is already determined by literal kind
}

void SemanticAnalyzer::check_identifier_expression(IdentifierExpression* expr) {
    auto symbol = lookup_symbol(expr->name);
    if (!symbol) {
        // Note: We no longer report "Undefined identifier" errors because they cause
        // too many false positives. Identifiers may be defined in C++1 passthrough code,
        // in other translation units, or via qualified names that we don't fully resolve.
        // This includes std:: names, Cpp2 builtins, and user-defined types/functions.
    } else {
        track_variable_usage(expr->name, expr->line);
    }
}

void SemanticAnalyzer::check_binary_expression(BinaryExpression* expr) {
    check_expression(expr->left.get());
    check_expression(expr->right.get());

    // Check for mixed-sign arithmetic
    if (expr->op == TokenType::Plus || expr->op == TokenType::Minus ||
        expr->op == TokenType::Asterisk || expr->op == TokenType::Slash) {
        check_mixed_sign_arithmetic(expr);
    }

    // Check for unsafe operations
    check_unsafe_operations(expr);
}

void SemanticAnalyzer::check_unary_expression(UnaryExpression* expr) {
    check_expression(expr->operand.get());
    check_unsafe_operations(expr);
}

void SemanticAnalyzer::check_call_expression(CallExpression* expr) {
    check_expression(expr->callee.get());

    // Check arguments (new structure)
    for (auto& arg : expr->arguments) {
        check_expression(arg.expr.get());
    }
    // Check legacy args for backward compat
    for (auto& arg : expr->args) {
        check_expression(arg.get());
    }

    // Resolve UFCS if necessary
    resolve_ufcs(expr);
}

void SemanticAnalyzer::check_member_access_expression(MemberAccessExpression* expr) {
    check_expression(expr->object.get());

    // Check if member exists in object type
    // This would require type information from the object
}

void SemanticAnalyzer::check_subscript_expression(SubscriptExpression* expr) {
    check_expression(expr->array.get());
    check_expression(expr->index.get());
    check_bounds_checking(expr);
}

void SemanticAnalyzer::check_ternary_expression(TernaryExpression* expr) {
    check_expression(expr->condition.get());
    check_expression(expr->then_expr.get());
    check_expression(expr->else_expr.get());
}

void SemanticAnalyzer::check_lambda_expression(LambdaExpression* expr) {
    push_scope();

    // Add parameters
    for (const auto& param : expr->parameters) {
        // Cpp2 allows parameters without explicit types - they become deduced/template parameters
        if (param.type) {
            check_type_ptr(param.type.get());
        }
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Parameter, param.name, param.type.get(), nullptr);
        add_symbol(param.name, std::move(symbol));
    }

    // Check body
    for (auto& stmt : expr->body) {
        check_statement(stmt.get());
    }

    pop_scope();
}

void SemanticAnalyzer::check_is_expression(IsExpression* expr) {
    check_expression(expr->expr.get());
    expr->type = check_type(std::move(expr->type));
}

void SemanticAnalyzer::check_as_expression(AsExpression* expr) {
    check_expression(expr->expr.get());
    expr->type = check_type(std::move(expr->type));
}

void SemanticAnalyzer::check_range_expression(RangeExpression* expr) {
    check_expression(expr->start.get());
    check_expression(expr->end.get());
}

std::unique_ptr<Type> SemanticAnalyzer::check_type(std::unique_ptr<Type> type) {
    if (!type) return nullptr;

    // Only change to Builtin if it's NOT a template type with arguments
    // Template types need to keep their kind to emit template arguments
    if ((type->kind == Type::Kind::UserDefined) &&
        is_builtin_type(type->name) && type->template_args.empty()) {
        type->kind = Type::Kind::Builtin;
    }

    // Note: We no longer report "Undefined type" errors here because they cause
    // too many false positives. Types may be defined in other translation units,
    // via C++1 passthrough, or with qualified names that we don't fully resolve.

    // Check template arguments
    for (auto& arg : type->template_args) {
        arg = check_type(std::move(arg));
    }

    return type;
}

void SemanticAnalyzer::check_type_ptr(const Type* type) {
    if (!type) return;

    // Note: We no longer report "Undefined type" errors here because they cause
    // too many false positives. Types may be defined in other translation units,
    // via C++1 passthrough, or with qualified names that we don't fully resolve.

    // Check template arguments (without modifying)
    for (const auto& arg : type->template_args) {
        check_type_ptr(arg.get());
    }
}

void SemanticAnalyzer::check_contract(ContractExpression* contract) {
    check_expression(contract->condition.get());
}

bool SemanticAnalyzer::is_type_compatible(const Type* lhs, const Type* rhs) const {
    // Simplified type compatibility check
    if (!lhs || !rhs) return false;
    if (lhs->kind == Type::Kind::Auto || rhs->kind == Type::Kind::Auto) return true;
    return lhs->name == rhs->name;
}

void SemanticAnalyzer::check_mixed_sign_arithmetic(BinaryExpression* expr) {
    // Check for signed/unsigned integer arithmetic
    // This would require actual type information
}

void SemanticAnalyzer::check_unsafe_operations(Expression* expr) {
    // Check for potentially unsafe operations
}

void SemanticAnalyzer::check_null_safety(Expression* expr) {
    // Check for null pointer dereferences
}

void SemanticAnalyzer::check_bounds_checking(SubscriptExpression* expr) {
    // Check array bounds
}

void SemanticAnalyzer::resolve_type_declaration(TypeDeclaration* type_decl) {
    if (!type_decl) return;

    push_scope();

    // Add template parameters as types in this scope
    for (const std::string& tparam : type_decl->template_parameters) {
        // Strip trailing ... for variadic parameters when creating symbol
        std::string param_name = tparam;
        if (param_name.size() >= 3 && param_name.substr(param_name.size() - 3) == "...") {
            param_name = param_name.substr(0, param_name.size() - 3);
        }
        auto template_type = std::make_unique<Type>(Type::Kind::UserDefined);
        template_type->name = param_name;
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Type, param_name, template_type.get(), nullptr);
        add_symbol(param_name, std::move(symbol));
        // Also add the pack expansion version (Ts...)
        if (tparam != param_name) {
            auto pack_type = std::make_unique<Type>(Type::Kind::UserDefined);
            pack_type->name = tparam;
            auto pack_symbol = std::make_unique<Symbol>(
                Symbol::Kind::Type, tparam, pack_type.get(), nullptr);
            add_symbol(tparam, std::move(pack_symbol));
        }
    }

    // Special handling for enum types: register enum members as qualified symbols
    if (type_decl->type_kind == TypeDeclaration::TypeKind::Enum) {
        resolve_enum_declaration(type_decl);
        pop_scope();
        return;
    }

    for (auto& member : type_decl->members) {
        check_declaration(member.get());
    }

    pop_scope();
}

void SemanticAnalyzer::resolve_enum_declaration(TypeDeclaration* enum_decl) {
    // Enum members don't need explicit types - they get the enum type
    // Register each member as a qualified symbol (EnumName::member) in the PARENT scope
    // so they're visible outside the enum definition
    auto parent_scope = current_scope->parent();
    if (!parent_scope) parent_scope = current_scope;  // fallback to current

    // First pass: collect all enum member names and register them
    // This allows initializers and member functions to reference other members
    std::vector<std::string> member_names;
    for (auto& member : enum_decl->members) {
        if (auto* var_member = dynamic_cast<VariableDeclaration*>(member.get())) {
            member_names.push_back(var_member->name);
            
            // Register qualified name (e.g., skat_game::clubs) in parent scope
            std::string qualified_name = enum_decl->name + "::" + var_member->name;
            auto enum_type = std::make_unique<Type>(Type::Kind::UserDefined);
            enum_type->name = enum_decl->name;
            auto symbol = std::make_unique<Symbol>(
                Symbol::Kind::EnumMember, qualified_name, enum_type.get(), var_member);
            parent_scope->add_symbol(qualified_name, std::move(symbol));
            
            // Also register unqualified name in current (enum) scope for internal use
            auto unqual_type = std::make_unique<Type>(Type::Kind::UserDefined);
            unqual_type->name = enum_decl->name;
            auto unqual_symbol = std::make_unique<Symbol>(
                Symbol::Kind::EnumMember, var_member->name, unqual_type.get(), var_member);
            add_symbol(var_member->name, std::move(unqual_symbol));
        }
    }

    // Second pass: check initializers and member functions
    // Now all enum members are in scope
    for (auto& member : enum_decl->members) {
        if (auto* var_member = dynamic_cast<VariableDeclaration*>(member.get())) {
            // Check initializer expression if present (can reference other members)
            if (var_member->initializer) {
                check_expression(var_member->initializer.get());
            }
        } else if (auto* func_member = dynamic_cast<FunctionDeclaration*>(member.get())) {
            // Enum member functions (like flip() in janus enum)
            // Register qualified name (e.g., janus::flip) in parent scope
            std::string qualified_name = enum_decl->name + "::" + func_member->name;
            auto func_type = std::make_unique<Type>(Type::Kind::Function);
            auto symbol = std::make_unique<Symbol>(
                Symbol::Kind::Function, qualified_name, func_type.get(), func_member);
            parent_scope->add_symbol(qualified_name, std::move(symbol));

            // Check function body - enum members are already in scope
            check_function(func_member);
        }
    }

    // Also register synthesized members for @enum (from_string, from_code, etc.) in parent scope
    const std::vector<std::string> synthesized_members = {
        "from_string", "from_code", "to_string", "to_code", "get_raw_value"
    };
    for (const auto& syn_member : synthesized_members) {
        std::string qualified_name = enum_decl->name + "::" + syn_member;
        auto func_type = std::make_unique<Type>(Type::Kind::Function);
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Function, qualified_name, func_type.get(), nullptr);
        parent_scope->add_symbol(qualified_name, std::move(symbol));
    }

    // For flag_enum, also register bitwise operation support  
    bool is_flag_enum = false;
    for (const auto& mf : enum_decl->metafunctions) {
        if (mf.find("flag_enum") != std::string::npos) {
            is_flag_enum = true;
            break;
        }
    }
    if (is_flag_enum) {
        const std::vector<std::string> flag_members = {"clear", "set"};
        for (const auto& flag_member : flag_members) {
            std::string qualified_name = enum_decl->name + "::" + flag_member;
            auto func_type = std::make_unique<Type>(Type::Kind::Function);
            auto symbol = std::make_unique<Symbol>(
                Symbol::Kind::Function, qualified_name, func_type.get(), nullptr);
            parent_scope->add_symbol(qualified_name, std::move(symbol));
        }
    }
}

void SemanticAnalyzer::resolve_ufcs(CallExpression* call) {
    if (!is_member_call(call->callee.get())) {
        // This could be a UFCS call
        // Check if first argument has the function as a member
    }
}

bool SemanticAnalyzer::is_member_call(Expression* callee) const {
    // Check if this is a member function call
    return callee->kind == Expression::Kind::MemberAccess;
}

void SemanticAnalyzer::push_scope() {
    current_scope = std::make_shared<Scope>(current_scope);
}

void SemanticAnalyzer::pop_scope() {
    current_scope = current_scope->parent();
}

void SemanticAnalyzer::add_symbol(const std::string& name, std::unique_ptr<Symbol> symbol) {
    current_scope->add_symbol(name, std::move(symbol));
}

Symbol* SemanticAnalyzer::lookup_symbol(const std::string& name) const {
    return current_scope ? current_scope->lookup(name) : nullptr;
}

bool SemanticAnalyzer::is_builtin_type(const std::string& name) const {
    // Handle C++ decltype(type) - this is a C++ construct, not a user-defined type
    if (name.size() >= 9 && name.substr(0, 9) == "decltype(") {
        return true;
    }
    static const std::unordered_set<std::string> builtins = {
        "bool", "char", "int", "int8", "int16", "int32", "int64",
        "uint", "uint8", "uint16", "uint32", "uint64",
        // Cpp2-style integer aliases
        "i8", "i16", "i32", "i64",
        "u8", "u16", "u32", "u64",
        "f32", "f64",
        "float", "double", "void", "auto", "string", "string_view",
        // C/C++ primitive types
        "unsigned", "signed", "long", "short", "size_t", "ssize_t",
        // Standard library types
        "std::string", "std::string_view", "std::ostream", "std::istream",
        "std::vector", "std::array", "std::span", "std::unique_ptr", "std::shared_ptr",
        "std::optional", "std::variant", "std::any", "std::tuple",
        "std::once_flag", "std::mutex", "std::thread", "std::atomic",
        "std::function", "std::pair", "std::map", "std::set", "std::unordered_map",
        "std::unordered_set", "std::list", "std::deque", "std::queue", "std::stack"
    };
    // Also accept std:: qualified names
    if (name.size() >= 5 && name.substr(0, 5) == "std::") {
        return true;
    }
    return builtins.contains(name);
}

void SemanticAnalyzer::register_builtin_types() {
    // Register built-in types in the symbol table
    push_scope(); // Built-in scope

    std::vector<std::string> builtin_names = {
        "bool", "char", "int", "int8", "int16", "int32", "int64",
        "uint", "uint8", "uint16", "uint32", "uint64",
        // Cpp2-style integer aliases
        "i8", "i16", "i32", "i64",
        "u8", "u16", "u32", "u64",
        "float", "double", "void", "auto", "string", "string_view"
    };

    for (const auto& name : builtin_names) {
        auto type = std::make_unique<Type>(Type::Kind::Builtin);
        type->name = name;
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Type, name, type.get(), nullptr);
        add_symbol(name, std::move(symbol));
        builtin_types[name] = std::move(type);
    }

    // Add std stubs
    {
        auto type = std::make_unique<Type>(Type::Kind::Builtin);
        type->name = "std::string";
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Type, "std::string", type.get(), nullptr);
        add_symbol("std::string", std::move(symbol));
        builtin_types["std::string"] = std::move(type);
    }
    {
        auto var_type = std::make_unique<Type>(Type::Kind::Builtin);
        var_type->name = "std::ostream";
        auto symbol = std::make_unique<Symbol>(
            Symbol::Kind::Variable, "std::cout", var_type.get(), nullptr);
        add_symbol("std::cout", std::move(symbol));
    }

    // Do not pop scope, so these remain as the root scope
    // pop_scope(); // Built-in scope
}

void SemanticAnalyzer::track_variable_usage(const std::string& name, std::size_t line) {
    auto it = variable_usage.find(name);
    if (it != variable_usage.end()) {
        it->second.used = true;
        it->second.last_use_line = line;
    }
}

void SemanticAnalyzer::check_unused_variables() {
    for (const auto& [name, usage] : variable_usage) {
        if (!usage.used) {
            report_warning(0, "Unused variable: " + name);
        }
    }
}

void SemanticAnalyzer::report_error(std::size_t line, const std::string& message) {
    errors.push_back(std::format("[line {}] Error: {}", line, message));
}

void SemanticAnalyzer::report_warning(std::size_t line, const std::string& message) {
    warnings.push_back(std::format("[line {}] Warning: {}", line, message));
}

// ============================================================================
// Concurrency Expression Checking (Kotlin-style structured concurrency)
// ============================================================================

void SemanticAnalyzer::check_await_expression(AwaitExpression* expr) {
    if (!expr) return;
    
    // Validate that await is only used inside a suspend function or coroutine scope
    if (!in_suspend_function && in_coroutine_scope_depth == 0) {
        report_error(expr->line, 
            "'await' can only be used inside a suspend function or coroutineScope");
    }
    
    // Check the awaited value
    check_expression(expr->value.get());
}

void SemanticAnalyzer::check_spawn_expression(SpawnExpression* expr) {
    if (!expr) return;
    
    // spawn should typically be within a coroutineScope for structured concurrency
    if (in_coroutine_scope_depth == 0) {
        report_warning(expr->line,
            "'launch' outside coroutineScope may result in unstructured concurrency");
    }
    
    // Check the task expression (typically a lambda or function call)
    check_expression(expr->task.get());
}

void SemanticAnalyzer::check_channel_send_expression(ChannelSendExpression* expr) {
    if (!expr) return;
    
    // Verify the channel exists in scope
    Symbol* channel_sym = lookup_symbol(expr->channel);
    if (!channel_sym) {
        report_error(expr->line, "Undeclared channel: " + expr->channel);
    }
    
    // Check the value being sent
    check_expression(expr->value.get());
}

void SemanticAnalyzer::check_channel_recv_expression(ChannelRecvExpression* expr) {
    if (!expr) return;
    
    // Verify the channel exists in scope
    Symbol* channel_sym = lookup_symbol(expr->channel);
    if (!channel_sym) {
        report_error(expr->line, "Undeclared channel: " + expr->channel);
    }
}

void SemanticAnalyzer::check_channel_select_expression(ChannelSelectExpression* expr) {
    if (!expr) return;
    
    // Check each case in the select
    for (const auto& case_ : expr->cases) {
        // Verify channel exists
        Symbol* channel_sym = lookup_symbol(case_.channel);
        if (!channel_sym) {
            report_error(expr->line, "Undeclared channel in select: " + case_.channel);
        }
        
        // Check the value (for send) and action expressions
        if (case_.value) check_expression(case_.value.get());
        if (case_.action) check_expression(case_.action.get());
    }
    
    // Check default case if present
    if (expr->default_case) {
        check_expression(expr->default_case.get());
    }
}

} // namespace cpp2_transpiler