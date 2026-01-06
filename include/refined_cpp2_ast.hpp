/*
 * Automatically generated from cppfront regression corpus
 * using Clang AST ground truth
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace cppfort {

enum class Qualifier {
    None,
    InOut,
    Out,
    Move,
    Forward
};

struct Parameter {
    std::string name;
    std::string type;
    std::vector<Qualifier> qualifiers;

    bool is_inout() const {
        for (auto q : qualifiers) {
            if (q == Qualifier::InOut) return true;
        }
        return false;
    }
};

struct FunctionDeclaration {
    std::string name;
    std::vector<Parameter> parameters;
    std::string return_type;
    std::unique_ptr<class Block> body;

    // Corpus-derived metadata
    bool is_nodiscard = false;
    bool is_template = false;
    std::vector<std::string> template_params;
};

// Function signatures extracted from corpus
/*
// From pure2-autodiff.mapping
// add: (b: double) -> double
// From pure2-assert-optional-not-null.mapping
// bad_optional_access: () -> int
// From pure2-assert-shared-ptr-not-null.mapping
// bad_shared_ptr_access: () -> int
// From pure2-assert-unique-ptr-not-null.mapping
// bad_unique_ptr_access: () -> int
// From pure2-regex_04_start_end.mapping
// create_result: (resultExpr: std::string) -> std::string
// From pure2-intro-example-hello-2022.mapping
// decorate: () -> int
// From pure2-autodiff.mapping
// direct_return: (x: double, y: double) -> double
// From pure2-bugfix-for-ufcs-arguments.mapping
// f: (_: t) -> i32
// From pure2-repeated-call.mapping
// f0: () -> _
// From pure2-repeated-call.mapping
// f1: () -> _
// From pure2-repeated-call.mapping
// f2: () -> _
// From pure2-repeated-call.mapping
// f3: () -> _
// From pure2-repeated-call.mapping
// f4: () -> _
// From pure2-assert-unique-ptr-not-null.mapping
// fine: () -> int
// From mixed-initialization-safety-3.mapping
// flip_a_coin: () -> bool
// From mixed-inspect-with-typeof-of-template-arg-list.mapping
// fun: (v : _) -> int
// From mixed-inspect-templates.mapping
// fun2: (v : _) -> std::string
// From pure2-types-down-upcast.mapping
// func_const: (b: B) -> void
// From pure2-types-down-upcast.mapping
// func_mut: (inout b: B) -> void
// From pure2-function-typeids.mapping
// g: () -> int
// From pure2-function-single-expression-body-default-return.mapping
// g2: () -> void
// From pure2-ufcs-member-access-and-chaining.mapping
// get_i: () -> int
// From mixed-bugfix-for-ufcs-non-local.mapping
// h: () -> t<o.f()>
// From pure2-function-typeids.mapping
// h_out: (s: std::string) -> std::string
// From mixed-inspect-values.mapping
// in_2_3: (x: int) -> bool
// From mixed-as-for-variant-20-types.mapping
// main: () -> int
// From pure2-hello.mapping
// name: () -> std::string
// From pure2-is-with-variable-and-value.mapping
// op_is: () -> bool
// From pure2-is-with-free-functions-predicate.mapping
// pred_: () -> bool
// From pure2-is-with-free-functions-predicate.mapping
// pred_d: (x : double) -> bool
// From pure2-is-with-free-functions-predicate.mapping
// pred_i: (x : int) -> bool
// From pure2-function-body-reflection.mapping
// prefix: () -> int
// From pure2-bugfix-for-ufcs-arguments.mapping
// print_res: (x: i32) -> i32
// From pure2-function-body-reflection.mapping
// qualified_ids: () -> _
// From pure2-print.mapping
// return: () -> std::string
// From pure2-regex_04_start_end.mapping
// sanitize: (copy str: std::string) -> std::string
// From mixed-postfix-expression-custom-formatting.mapping
// test: () -> std::string
// From pure2-contracts.mapping
// test_condition_evaluation: () -> bool
// From pure2-union.mapping
// to_string: () -> std::string
// From pure2-ufcs-member-access-and-chaining.mapping
// ufcs: () -> int

 */

// Statement types
class Statement {
public:
    enum class Kind {
        Return,
        Expression,
        VariableDeclaration,
        Block
    };

    virtual ~Statement() = default;
    virtual Kind getKind() const = 0;
};

class ReturnStatement : public Statement {
public:
    std::unique_ptr<class Expression> value;

    Kind getKind() const override { return Kind::Return; }
};

class ExpressionStatement : public Statement {
public:
    std::unique_ptr<class Expression> expr;

    Kind getKind() const override { return Kind::Expression; }
};

class VariableDeclaration : public Statement {
public:
    std::string name;
    std::string type;
    std::unique_ptr<Expression> initializer;
    std::vector<Qualifier> qualifiers;

    Kind getKind() const override { return Kind::VariableDeclaration; }
};

class Block : public Statement {
public:
    std::vector<std::unique_ptr<Statement>> statements;

    Kind getKind() const override { return Kind::Block; }
};

// Expression types
class Expression {
public:
    enum class Kind {
        Literal,
        Identifier,
        FunctionCall,
        BinaryOp,
        UnaryOp,
        UFCS
    };

    virtual ~Expression() = default;
    virtual Kind getKind() const = 0;
};

class FunctionCallExpression : public Expression {
public:
    std::string function_name;
    std::vector<std::unique_ptr<Expression>> arguments;
    bool is_ufcs = false;  // Unified Function Call Syntax

    Kind getKind() const override { return Kind::FunctionCall; }
};

// High-frequency patterns from corpus
class BoundsCheckExpression : public Expression {
public:
    std::unique_ptr<Expression> container;
    std::unique_ptr<Expression> index;

    Kind getKind() const override { return Kind::UnaryOp; }  // Special unary op
};

} // namespace cppfort
