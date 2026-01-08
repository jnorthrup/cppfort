// include/codegen/declaration_emitter.hpp - Declaration generation for C++ code
// Part of Phase 2: Code Generator Extraction
#pragma once

#include "codegen/emitter_context.hpp"
#include "codegen/statement_emitter.hpp"
#include "codegen/expression_emitter.hpp"
#include "codegen/type_emitter.hpp"
#include "ast.hpp"

namespace cpp2_transpiler {

/// Generates C++ declarations from Cpp2 AST declarations
class DeclarationEmitter : public EmitterBase {
public:
    DeclarationEmitter(EmitterContext& ctx, 
                       StatementEmitter& stmt_emitter,
                       ExpressionEmitter& expr_emitter, 
                       TypeEmitter& type_emitter)
        : EmitterBase(ctx), 
          stmt_emitter_(stmt_emitter),
          expr_emitter_(expr_emitter), 
          type_emitter_(type_emitter) {}

    /// Generate declaration to output stream
    void generate_declaration(Declaration* decl);
    
    /// Generate forward declaration (for functions)
    void generate_function_forward_declaration(FunctionDeclaration* decl);

private:
    void generate_variable_declaration(VariableDeclaration* decl);
    void generate_function_declaration(FunctionDeclaration* decl);
    void generate_special_member_function(FunctionDeclaration* decl);
    void generate_operator_eq_colon(OperatorDeclaration* decl);
    void generate_type_declaration(TypeDeclaration* decl);
    void generate_namespace_declaration(NamespaceDeclaration* decl);
    void generate_operator_declaration(OperatorDeclaration* decl);
    void generate_using_declaration(UsingDeclaration* decl);
    void generate_import_declaration(ImportDeclaration* decl);
    void generate_cpp1_passthrough_declaration(Cpp1PassthroughDeclaration* decl);

    // Metafunction expansion helpers
    void expand_metafunction(TypeDeclaration* type_decl);
    void expand_value_metafunction(TypeDeclaration* type_decl);
    void expand_ordered_metafunction(TypeDeclaration* type_decl);
    void expand_copyable_metafunction(TypeDeclaration* type_decl);
    void expand_interface_metafunction(TypeDeclaration* type_decl);
    void expand_polymorphic_base_metafunction(TypeDeclaration* type_decl);
    void expand_enum_metafunction(TypeDeclaration* type_decl);
    void expand_flag_enum_metafunction(TypeDeclaration* type_decl);
    void expand_union_metafunction(TypeDeclaration* type_decl);
    void expand_struct_metafunction(TypeDeclaration* type_decl);
    void expand_hashable_metafunction(TypeDeclaration* type_decl);

    // Utility
    bool needs_nodiscard(FunctionDeclaration* func);
    std::string generate_template_param(const std::string& param);

    StatementEmitter& stmt_emitter_;
    ExpressionEmitter& expr_emitter_;
    TypeEmitter& type_emitter_;
};

} // namespace cpp2_transpiler
