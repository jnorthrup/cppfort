#include "metafunction_processor.hpp"
#include <iostream>

namespace cpp2_transpiler {

MetafunctionProcessor::MetafunctionProcessor() {
    register_builtin_metafunctions();
}

void MetafunctionProcessor::process(AST& ast) {
    for (auto& decl : ast.declarations) {
        if (auto type_decl = dynamic_cast<TypeDeclaration*>(decl.get())) {
            // Process metafunctions on this type
            for (const auto& meta_name : type_decl->metafunctions) {
                auto it = metafunctions.find(meta_name);
                if (it != metafunctions.end()) {
                    it->second.processor(type_decl);
                } else {
                    std::cerr << "Warning: Unknown metafunction @" << meta_name << std::endl;
                }
            }
        }
    }
}

void MetafunctionProcessor::register_builtin_metafunctions() {
    // Register all built-in metafunctions
    metafunctions["value"] = {
        "value",
        {},
        [this](TypeDeclaration* type_decl) { process_value(type_decl); }
    };

    metafunctions["ordered"] = {
        "ordered",
        {},
        [this](TypeDeclaration* type_decl) { process_ordered(type_decl); }
    };

    metafunctions["copyable"] = {
        "copyable",
        {},
        [this](TypeDeclaration* type_decl) { process_copyable(type_decl); }
    };

    metafunctions["interface"] = {
        "interface",
        {},
        [this](TypeDeclaration* type_decl) { process_interface(type_decl); }
    };

    metafunctions["polymorphic_base"] = {
        "polymorphic_base",
        {},
        [this](TypeDeclaration* type_decl) { process_polymorphic_base(type_decl); }
    };

    metafunctions["enum"] = {
        "enum",
        {},
        [this](TypeDeclaration* type_decl) { process_enum(type_decl); }
    };

    metafunctions["flag_enum"] = {
        "flag_enum",
        {},
        [this](TypeDeclaration* type_decl) { process_flag_enum(type_decl); }
    };

    metafunctions["union"] = {
        "union",
        {},
        [this](TypeDeclaration* type_decl) { process_union(type_decl); }
    };

    metafunctions["struct"] = {
        "struct",
        {},
        [this](TypeDeclaration* type_decl) { process_struct(type_decl); }
    };

    metafunctions["hashable"] = {
        "hashable",
        {},
        [this](TypeDeclaration* type_decl) { process_hashable(type_decl); }
    };
}

void MetafunctionProcessor::process_value(TypeDeclaration* type_decl) {
    // Generate value semantics: default constructor, copy/move operations, assignment
    generate_copy_operations(type_decl);
    generate_move_operations(type_decl);
}

void MetafunctionProcessor::process_ordered(TypeDeclaration* type_decl) {
    // Generate comparison operators
    generate_comparison_operators(type_decl);
}

void MetafunctionProcessor::process_copyable(TypeDeclaration* type_decl) {
    // Ensure copy operations are available
    generate_copy_operations(type_decl);
}

void MetafunctionProcessor::process_interface(TypeDeclaration* type_decl) {
    // Convert to abstract base class with pure virtual functions
    type_decl->type_kind = TypeDeclaration::TypeKind::Class;

    // Add virtual destructor
    auto destructor = std::make_unique<FunctionDeclaration>("~" + type_decl->name, 0);
    destructor->is_virtual = true;
    destructor->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    destructor->return_type->name = "void";
    add_member_to_type(type_decl, std::move(destructor));
}

void MetafunctionProcessor::process_polymorphic_base(TypeDeclaration* type_decl) {
    // Add virtual functions for polymorphic behavior
    type_decl->type_kind = TypeDeclaration::TypeKind::Class;

    // Add virtual destructor
    auto destructor = std::make_unique<FunctionDeclaration>("~" + type_decl->name, 0);
    destructor->is_virtual = true;
    destructor->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    destructor->return_type->name = "void";
    add_member_to_type(type_decl, std::move(destructor));

    // Add virtual clone function
    auto clone = std::make_unique<FunctionDeclaration>("clone", 0);
    clone->is_virtual = true;
    clone->return_type = std::make_unique<Type>(Type::Kind::Pointer);
    clone->return_type->name = type_decl->name + "*";
    add_member_to_type(type_decl, std::move(clone));
}

void MetafunctionProcessor::process_enum(TypeDeclaration* type_decl) {
    // Convert to enum with underlying type
    type_decl->type_kind = TypeDeclaration::TypeKind::Enum;

    if (!type_decl->underlying_type) {
        type_decl->underlying_type = std::make_unique<Type>(Type::Kind::Builtin);
        type_decl->underlying_type->name = "int";
    }
}

void MetafunctionProcessor::process_flag_enum(TypeDeclaration* type_decl) {
    // Process as enum but add bitwise operators
    process_enum(type_decl);

    // Add bitwise operators
    add_operator(type_decl, "&", {type_decl->name, type_decl->name}, type_decl->name);
    add_operator(type_decl, "|", {type_decl->name, type_decl->name}, type_decl->name);
    add_operator(type_decl, "^", {type_decl->name, type_decl->name}, type_decl->name);
    add_operator(type_decl, "~", {type_decl->name}, type_decl->name);
}

void MetafunctionProcessor::process_union(TypeDeclaration* type_decl) {
    // Convert to union type
    type_decl->type_kind = TypeDeclaration::TypeKind::Union;
}

void MetafunctionProcessor::process_struct(TypeDeclaration* type_decl) {
    // Ensure it's a struct type
    type_decl->type_kind = TypeDeclaration::TypeKind::Struct;
}

void MetafunctionProcessor::process_hashable(TypeDeclaration* type_decl) {
    // Generate hash function
    generate_hash_function(type_decl);
}

void MetafunctionProcessor::generate_comparison_operators(TypeDeclaration* type_decl) {
    // Generate operator==
    add_operator(type_decl, "==", {type_decl->name, type_decl->name}, "bool");

    // Generate operator!=
    add_operator(type_decl, "!=", {type_decl->name, type_decl->name}, "bool");

    // Generate operator<
    add_operator(type_decl, "<", {type_decl->name, type_decl->name}, "bool");

    // Generate operator<=
    add_operator(type_decl, "<=", {type_decl->name, type_decl->name}, "bool");

    // Generate operator>
    add_operator(type_decl, ">", {type_decl->name, type_decl->name}, "bool");

    // Generate operator>=
    add_operator(type_decl, ">=", {type_decl->name, type_decl->name}, "bool");
}

void MetafunctionProcessor::generate_copy_operations(TypeDeclaration* type_decl) {
    // Generate copy constructor
    auto copy_ctor = std::make_unique<FunctionDeclaration>(type_decl->name, 0);
    copy_ctor->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    copy_ctor->return_type->name = "void";

    FunctionDeclaration::Parameter param;
    param.name = "other";
    param.type = std::make_unique<Type>(Type::Kind::Reference);
    param.type->name = "const " + type_decl->name + "&";
    copy_ctor->parameters.push_back(std::move(param));

    add_member_to_type(type_decl, std::move(copy_ctor));

    // Generate copy assignment operator
    add_operator(type_decl, "=", {type_decl->name, "const " + type_decl->name + "&"}, type_decl->name + "&");
}

void MetafunctionProcessor::generate_move_operations(TypeDeclaration* type_decl) {
    // Generate move constructor
    auto move_ctor = std::make_unique<FunctionDeclaration>(type_decl->name, 0);
    move_ctor->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    move_ctor->return_type->name = "void";

    FunctionDeclaration::Parameter param;
    param.name = "other";
    param.type = std::make_unique<Type>(Type::Kind::Reference);
    param.type->name = type_decl->name + "&&";
    move_ctor->parameters.push_back(std::move(param));

    add_member_to_type(type_decl, std::move(move_ctor));

    // Generate move assignment operator
    add_operator(type_decl, "=", {type_decl->name, type_decl->name + "&&"}, type_decl->name + "&");
}

void MetafunctionProcessor::generate_hash_function(TypeDeclaration* type_decl) {
    auto hash_func = std::make_unique<FunctionDeclaration>("hash", 0);
    hash_func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    hash_func->return_type->name = "size_t";

    FunctionDeclaration::Parameter param;
    param.name = "value";
    param.type = std::make_unique<Type>(Type::Kind::Reference);
    param.type->name = "const " + type_decl->name + "&";
    hash_func->parameters.push_back(std::move(param));

    add_member_to_type(type_decl, std::move(hash_func));
}

bool MetafunctionProcessor::has_metafunction(TypeDeclaration* type_decl, const std::string& name) const {
    return std::find(type_decl->metafunctions.begin(), type_decl->metafunctions.end(), name) !=
           type_decl->metafunctions.end();
}

std::vector<std::string> MetafunctionProcessor::get_metafunction_args(
    TypeDeclaration* type_decl, const std::string& name) const {
    // In a real implementation, this would parse metafunction arguments
    return {};
}

void MetafunctionProcessor::add_member_to_type(TypeDeclaration* type_decl, std::unique_ptr<Declaration> member) {
    type_decl->members.push_back(std::move(member));
}

void MetafunctionProcessor::add_constructor(TypeDeclaration* type_decl, const std::vector<std::string>& parameters) {
    auto ctor = std::make_unique<FunctionDeclaration>(type_decl->name, 0);
    ctor->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    ctor->return_type->name = "void";

    for (const auto& param : parameters) {
        FunctionDeclaration::Parameter p;
        p.name = param;
        p.type = std::make_unique<Type>(Type::Kind::Auto);
        ctor->parameters.push_back(std::move(p));
    }

    add_member_to_type(type_decl, std::move(ctor));
}

void MetafunctionProcessor::add_operator(TypeDeclaration* type_decl, const std::string& op,
                                        const std::vector<std::string>& params, const std::string& return_type) {
    auto op_decl = std::make_unique<OperatorDeclaration>("operator" + op, 0);
    op_decl->return_type = std::make_unique<Type>(Type::Kind::UserDefined);
    op_decl->return_type->name = return_type;

    for (const auto& param : params) {
        auto p = std::make_unique<FunctionDeclaration::Parameter>();
        p->name = "param";
        p->type = std::make_unique<Type>(Type::Kind::UserDefined);
        p->type->name = param;
        op_decl->parameters.push_back(std::move(p));
    }

    add_member_to_type(type_decl, std::move(op_decl));
}

} // namespace cpp2_transpiler