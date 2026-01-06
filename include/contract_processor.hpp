#pragma once

#include "ast.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace cpp2_transpiler {

class ContractProcessor {
public:
    ContractProcessor();
    void process(AST& ast);

private:
    struct ContractGroup {
        std::string name;
        std::vector<std::unique_ptr<ContractExpression>> preconditions;
        std::vector<std::unique_ptr<ContractExpression>> postconditions;
        std::vector<std::unique_ptr<ContractExpression>> assertions;
        std::string violation_handler;
    };

    std::unordered_map<std::string, ContractGroup> contract_groups;

    // Processing methods
    void process_declaration(Declaration* decl);
    void process_function(FunctionDeclaration* func);
    void process_contract(ContractExpression* contract);

    // Contract group management
    void create_contract_group(const std::string& function_name, FunctionDeclaration* func);
    void add_to_contract_group(const std::string& group_name, std::unique_ptr<ContractExpression> contract);
    std::string get_contract_group_name(const std::string& function_name);

    // Code generation helpers
    void generate_contract_check(ContractExpression* contract, const std::string& group_name);
    void generate_precondition_check(ContractExpression* contract, const std::string& group_name);
    void generate_postcondition_check(ContractExpression* contract, const std::string& group_name);
    void generate_assertion_check(ContractExpression* contract, const std::string& group_name);

    // Contract evaluation
    std::string generate_contract_predicate(ContractExpression* contract);
    std::string generate_contract_message(ContractExpression* contract);

    // Contract violation handling
    void generate_contract_violation_handler(const std::string& group_name);
    void inject_contract_violation_check(FunctionDeclaration* func);

    // Capture handling
    void process_contract_captures(ContractExpression* contract, FunctionDeclaration* func);
    std::vector<std::string> extract_captured_variables(ContractExpression* contract);

    // Utility methods
    bool is_contract_expression(Statement* stmt) const;
    ContractExpression* get_contract_expression(Statement* stmt) const;
    std::string generate_unique_temporary_name(const std::string& base);
};

} // namespace cpp2_transpiler