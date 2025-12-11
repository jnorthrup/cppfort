#pragma once

#include "ast.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace cpp2_transpiler {

class MetafunctionProcessor {
public:
    MetafunctionProcessor();
    void process(AST& ast);

private:
    struct Metafunction {
        std::string name;
        std::vector<std::string> parameters;
        std::function<void(TypeDeclaration*)> processor;
    };

    std::unordered_map<std::string, Metafunction> metafunctions;

    // Built-in metafunctions
    void register_builtin_metafunctions();

    // Metafunction implementations
    void process_value(TypeDeclaration* type_decl);
    void process_ordered(TypeDeclaration* type_decl);
    void process_copyable(TypeDeclaration* type_decl);
    void process_interface(TypeDeclaration* type_decl);
    void process_polymorphic_base(TypeDeclaration* type_decl);
    void process_enum(TypeDeclaration* type_decl);
    void process_flag_enum(TypeDeclaration* type_decl);
    void process_union(TypeDeclaration* type_decl);
    void process_struct(TypeDeclaration* type_decl);
    void process_hashable(TypeDeclaration* type_decl);

    // Generation helpers
    void generate_comparison_operators(TypeDeclaration* type_decl);
    void generate_copy_operations(TypeDeclaration* type_decl);
    void generate_move_operations(TypeDeclaration* type_decl);
    void generate_hash_function(TypeDeclaration* type_decl);
    void generate_string_conversion(TypeDeclaration* type_decl);
    void generate_iterator_functions(TypeDeclaration* type_decl);

    // Utility methods
    bool has_metafunction(TypeDeclaration* type_decl, const std::string& name) const;
    std::vector<std::string> get_metafunction_args(TypeDeclaration* type_decl, const std::string& name) const;
    void add_member_to_type(TypeDeclaration* type_decl, std::unique_ptr<Declaration> member);
    void add_constructor(TypeDeclaration* type_decl, const std::vector<std::string>& parameters);
    void add_operator(TypeDeclaration* type_decl, const std::string& op,
                     const std::vector<std::string>& params, const std::string& return_type);
};

} // namespace cpp2_transpiler