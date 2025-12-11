#include "contract_processor.hpp"
#include <iostream>

namespace cpp2_transpiler {

ContractProcessor::ContractProcessor() {}

void ContractProcessor::process(AST& ast) {
    // First pass: collect all contracts and create contract groups
    for (auto& decl : ast.declarations) {
        if (auto func = dynamic_cast<FunctionDeclaration*>(decl.get())) {
            create_contract_group(func->name, func);
        }
    }

    // Second pass: process all declarations
    for (auto& decl : ast.declarations) {
        process_declaration(decl.get());
    }

    // Generate contract violation handlers
    for (const auto& [name, group] : contract_groups) {
        generate_contract_violation_handler(name);
    }
}

void ContractProcessor::process_declaration(Declaration* decl) {
    if (!decl) return;

    switch (decl->kind) {
        case Declaration::Kind::Function:
            process_function(static_cast<FunctionDeclaration*>(decl));
            break;
        case Declaration::Kind::Namespace: {
            auto ns = static_cast<NamespaceDeclaration*>(decl);
            for (auto& member : ns->members) {
                process_declaration(member.get());
            }
            break;
        }
        default:
            // Process contracts in other declaration types if needed
            break;
    }
}

void ContractProcessor::process_function(FunctionDeclaration* func) {
    if (!func) return;

    // Extract contract statements from function body
    std::vector<ContractExpression*> contracts;

    if (func->body) {
        if (auto block = dynamic_cast<BlockStatement*>(func->body.get())) {
            for (auto it = block->statements.begin(); it != block->statements.end();) {
                if (is_contract_expression(it->get())) {
                    auto contract_expr = get_contract_expression(it->get());
                    if (contract_expr) {
                        contracts.push_back(contract_expr);
                        process_contract(contract_expr);
                    }
                    it = block->statements.erase(it); // Remove from original body
                } else {
                    ++it;
                }
            }
        }
    }

    // Add contracts to the function's contract group
    std::string group_name = get_contract_group_name(func->name);
    for (auto contract : contracts) {
        add_to_contract_group(group_name, contract);
    }

    // Inject contract checks into function body
    if (!contracts.empty()) {
        inject_contract_violation_check(func);
    }
}

void ContractProcessor::process_contract(ContractExpression* contract) {
    if (!contract) return;

    // Process contract captures
    // This would involve analyzing the contract expression for variable references

    // Generate contract predicate
    std::string predicate = generate_contract_predicate(contract);

    // Generate contract message
    std::string message = generate_contract_message(contract);
}

void ContractProcessor::create_contract_group(const std::string& function_name, FunctionDeclaration* func) {
    std::string group_name = get_contract_group_name(function_name);

    if (contract_groups.find(group_name) == contract_groups.end()) {
        ContractGroup group;
        group.name = group_name;
        group.violation_handler = "contract_violation_" + group_name;
        contract_groups[group_name] = std::move(group);
    }
}

void ContractProcessor::add_to_contract_group(const std::string& group_name, ContractExpression* contract) {
    auto it = contract_groups.find(group_name);
    if (it != contract_groups.end()) {
        switch (contract->kind) {
            case ContractExpression::ContractKind::Pre:
                it->second.preconditions.push_back(contract);
                break;
            case ContractExpression::ContractKind::Post:
                it->second.postconditions.push_back(contract);
                break;
            case ContractExpression::ContractKind::Assert:
                it->second.assertions.push_back(contract);
                break;
        }
    }
}

std::string ContractProcessor::get_contract_group_name(const std::string& function_name) {
    return "contracts_" + function_name;
}

void ContractProcessor::generate_contract_check(ContractExpression* contract, const std::string& group_name) {
    switch (contract->kind) {
        case ContractExpression::ContractKind::Pre:
            generate_precondition_check(contract, group_name);
            break;
        case ContractExpression::ContractKind::Post:
            generate_postcondition_check(contract, group_name);
            break;
        case ContractExpression::ContractKind::Assert:
            generate_assertion_check(contract, group_name);
            break;
    }
}

void ContractProcessor::generate_precondition_check(ContractExpression* contract, const std::string& group_name) {
    // In a real implementation, this would generate code like:
    // if (!(contract->condition)) {
    //     contract_violation_handler(group_name, "Precondition failed", ...);
    // }
}

void ContractProcessor::generate_postcondition_check(ContractExpression* contract, const std::string& group_name) {
    // Generate postcondition check with return value capture
}

void ContractProcessor::generate_assertion_check(ContractExpression* contract, const std::string& group_name) {
    // Generate assertion check
}

std::string ContractProcessor::generate_contract_predicate(ContractExpression* contract) {
    // Convert contract expression to C++ predicate
    return "true"; // Simplified
}

std::string ContractProcessor::generate_contract_message(ContractExpression* contract) {
    if (contract->message) {
        return *contract->message;
    }
    return "Contract violation";
}

void ContractProcessor::generate_contract_violation_handler(const std::string& group_name) {
    // Generate a function to handle contract violations for this group
    // This would typically terminate the program or throw an exception
}

void ContractProcessor::inject_contract_violation_check(FunctionDeclaration* func) {
    if (!func || !func->body) return;

    std::string group_name = get_contract_group_name(func->name);
    auto it = contract_groups.find(group_name);
    if (it == contract_groups.end() || it->second.preconditions.empty()) {
        return;
    }

    // Create a new block for the function with precondition checks
    auto new_block = std::make_unique<BlockStatement>(func->line);

    // Add precondition checks at the beginning
    for (auto precond : it->second.preconditions) {
        auto assert_stmt = std::make_unique<ContractStatement>(std::make_unique<ContractExpression>(*precond), precond->line);
        new_block->statements.push_back(std::move(assert_stmt));
    }

    // Add original function body
    if (auto old_block = dynamic_cast<BlockStatement*>(func->body.get())) {
        for (auto& stmt : old_block->statements) {
            new_block->statements.push_back(std::move(stmt));
        }
    } else {
        new_block->statements.push_back(std::move(func->body));
    }

    // Replace function body
    func->body = std::move(new_block);
}

void ContractProcessor::process_contract_captures(ContractExpression* contract, FunctionDeclaration* func) {
    // Extract and process variables captured by the contract
    auto captured = extract_captured_variables(contract);
    // Add these to the contract's capture list
}

std::vector<std::string> ContractProcessor::extract_captured_variables(ContractExpression* contract) {
    // Analyze the contract expression to find referenced variables
    return {}; // Simplified
}

bool ContractProcessor::is_contract_expression(Statement* stmt) const {
    return stmt && stmt->kind == Statement::Kind::Contract;
}

ContractExpression* ContractProcessor::get_contract_expression(Statement* stmt) const {
    if (is_contract_expression(stmt)) {
        return static_cast<ContractStatement*>(stmt)->contract.get();
    }
    return nullptr;
}

std::string ContractProcessor::generate_unique_temporary_name(const std::string& base) {
    static int counter = 0;
    return base + std::to_string(counter++);
}

} // namespace cpp2_transpiler