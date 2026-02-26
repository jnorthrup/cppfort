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

static FunctionDeclaration::Parameter make_param(
    const std::string& name,
    std::initializer_list<ParameterQualifier> qualifiers,
    bool with_type = true) {
    FunctionDeclaration::Parameter param;
    param.name = name;
    if (with_type) {
        param.type = make_int_type();
    }
    for (auto q : qualifiers) {
        param.qualifiers.push_back(q);
    }
    return param;
}

static FunctionDeclaration* build_test_function(AST& ast) {
    auto func = std::make_unique<FunctionDeclaration>("sem_params", 42);
    func->return_type = make_int_type();
    func->parameters.push_back(make_param("a", {ParameterQualifier::InOut}));
    func->parameters.push_back(make_param("b", {ParameterQualifier::Out}));
    func->parameters.push_back(make_param("c", {ParameterQualifier::Move}));
    func->parameters.push_back(make_param("d", {}));
    func->parameters.push_back(make_param("e", {ParameterQualifier::Forward}, false)); // generic/deduced
    func->parameters.push_back(make_param("f", {ParameterQualifier::Out, ParameterQualifier::Move}));

    auto* func_ptr = func.get();
    ast.declarations.push_back(std::move(func));
    return func_ptr;
}

int main() {
    AST ast;
    auto* func = build_test_function(ast);

    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);

    assert(func->parameters.size() == 6);

    const auto& inout = func->parameters[0];
    assert(inout.semantic_info != nullptr);
    assert(inout.semantic_info->parameter.has_value());
    assert(inout.semantic_info->parameter->raw_qualifier == ParameterQualifier::InOut);
    assert(inout.semantic_info->parameter->effective_qualifier == ParameterQualifier::InOut);
    assert(inout.semantic_info->borrow.kind == OwnershipKind::MutBorrowed);

    const auto& out = func->parameters[1];
    assert(out.semantic_info->parameter.has_value());
    assert(out.semantic_info->parameter->write_before_return_expected);
    assert(out.semantic_info->parameter->mutable_access_expected);

    const auto& move = func->parameters[2];
    assert(move.semantic_info->parameter.has_value());
    assert(move.semantic_info->parameter->move_transfer_expected);
    assert(move.semantic_info->borrow.kind == OwnershipKind::Moved);

    const auto& unqualified = func->parameters[3];
    assert(unqualified.semantic_info->parameter.has_value());
    assert(!unqualified.semantic_info->parameter->has_explicit_qualifier);
    assert(unqualified.semantic_info->parameter->raw_qualifier == ParameterQualifier::None);
    assert(unqualified.semantic_info->parameter->effective_qualifier == ParameterQualifier::None);
    assert(unqualified.semantic_info->borrow.kind == OwnershipKind::Owned);

    const auto& generic = func->parameters[4];
    assert(generic.type == nullptr);
    assert(generic.semantic_info->parameter.has_value());
    assert(generic.semantic_info->parameter->raw_qualifier == ParameterQualifier::Forward);
    assert(generic.semantic_info->borrow.kind == OwnershipKind::Moved);

    const auto& multi = func->parameters[5];
    assert(multi.semantic_info->parameter.has_value());
    assert(multi.semantic_info->parameter->raw_qualifier == ParameterQualifier::Out);
    assert(multi.semantic_info->parameter->effective_qualifier == ParameterQualifier::Out);
    assert(multi.semantic_info->borrow.kind == OwnershipKind::MutBorrowed);

    std::cout << "semantic_parameter_metadata_test passed\n";
    return 0;
}
