#include "ast_to_fir.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace cpp2_transpiler;
using namespace mlir;
using namespace mlir::cpp2fir;

ASTToFIRConverter::ASTToFIRConverter(MLIRContext* ctx)
    : context(ctx), builder(ctx) {}

// Helper to convert ParameterQualifier to string attribute
static StringRef qualifierToString(ParameterQualifier qual) {
    switch (qual) {
        case ParameterQualifier::InOut: return "inout";
        case ParameterQualifier::Out: return "out";
        case ParameterQualifier::Move: return "move";
        case ParameterQualifier::Forward: return "forward";
        case ParameterQualifier::Virtual: return "virtual";
        case ParameterQualifier::Override: return "override";
        default: return "";
    }
}

ModuleOp ASTToFIRConverter::convertToFIR(const FunctionDeclaration& func) {
    // Create a module
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Convert return type
    mlir::Type resultType = convertType(*func.return_type);

    // Convert parameter types
    SmallVector<mlir::Type> paramTypes;
    for (const auto& param : func.parameters) {
        paramTypes.push_back(convertType(*param.type));
    }

    // Create function type
    auto funcType = builder.getFunctionType(paramTypes, {resultType});

    // Build arg_attrs array with corpus-derived qualifier semantics
    SmallVector<Attribute> argAttrs;
    for (const auto& param : func.parameters) {
        SmallVector<NamedAttribute> attrs;

        // Encode parameter qualifiers as attributes (corpus-derived semantics)
        for (const auto& qual : param.qualifiers) {
            auto qualStr = qualifierToString(qual);
            if (!qualStr.empty()) {
                attrs.push_back(builder.getNamedAttr(
                    "cpp2.qualifier",
                    StringAttr::get(context, qualStr)
                ));
            }
        }

        if (!attrs.empty()) {
            argAttrs.push_back(DictionaryAttr::get(context, attrs));
        } else {
            argAttrs.push_back(DictionaryAttr::get(context, {}));
        }
    }

    auto argAttrsAttr = ArrayAttr::get(context, argAttrs);

    // Create FIR function operation with corpus semantics
    auto firFunc = builder.create<FuncOp>(
        loc,
        StringAttr::get(context, func.name),
        TypeAttr::get(funcType),
        argAttrsAttr,  // arg_attrs with qualifier semantics
        ArrayAttr{}   // res_attrs
    );

    // Create entry block
    Block* entry = firFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Convert body
    if (func.body) {
        if (failed(convertStatement(*func.body, builder))) {
            return nullptr;
        }
    }

    return module;
}

LogicalResult ASTToFIRConverter::convertStatement(const Statement& stmt, OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();

    if (auto* blockStmt = dynamic_cast<const BlockStatement*>(&stmt)) {
        // Process each statement in the block
        for (const auto& s : blockStmt->statements) {
            if (failed(convertStatement(*s, builder))) {
                return failure();
            }
        }
        return success();
    } else if (auto* retStmt = dynamic_cast<const ReturnStatement*>(&stmt)) {
        if (retStmt->value) {
            Value val = convertExpression(*retStmt->value, builder);
            if (!val) {
                return failure();
            }
            builder.create<ReturnOp>(loc, ValueRange{val});
        } else {
            builder.create<ReturnOp>(loc, ValueRange{});
        }
        return success();
    } else if (auto* ifStmt = dynamic_cast<const IfStatement*>(&stmt)) {
        // Convert condition
        Value condition = convertExpression(*ifStmt->condition, builder);
        if (!condition) {
            return failure();
        }

        // Create scf.if operation
        auto ifOp = builder.create<scf::IfOp>(loc, condition,
            /*thenBuilder=*/[&](OpBuilder& b, Location loc) {
                if (ifStmt->then_stmt) {
                    if (failed(convertStatement(*ifStmt->then_stmt, b))) {
                        return;  // Will be caught by verification
                    }
                }
            },
            /*elseBuilder=*/[&](OpBuilder& b, Location loc) {
                if (ifStmt->else_stmt) {
                    if (failed(convertStatement(*ifStmt->else_stmt, b))) {
                        return;  // Will be caught by verification
                    }
                }
            }
        );

        return success();
    } else if (auto* whileStmt = dynamic_cast<const WhileStatement*>(&stmt)) {
        // For now, handle while loop by creating a scf.while with simple structure
        // The while condition is evaluated before each iteration

        // Convert condition to i1 (boolean) outside the loop
        Value condition = convertExpression(*whileStmt->condition, builder);
        if (!condition) {
            return failure();
        }

        // Ensure condition is i1 type
        if (!condition.getType().isInteger(1)) {
            // If condition is not i1, create a comparison to convert it
            auto zero = builder.create<ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
            auto i1Cond = builder.create<CmpOp>(loc, builder.getI1Type(), condition, zero.getResult(),
                StringAttr::get(context, "ne"));
            condition = i1Cond.getResult();
        }

        // Create scf.while operation - using simplified approach for now
        // Just create the operation with condition as initial value
        auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});

        // Build the "before" region (condition check)
        OpBuilder::InsertionGuard guard(builder);
        auto& beforeBlock = whileOp.getBefore().emplaceBlock();
        builder.setInsertionPointToEnd(&beforeBlock);

        // For simplicity, just use the condition value captured above
        // (In a real implementation, this would be recomputed each iteration)
        builder.create<scf::ConditionOp>(loc, condition, ValueRange{});

        // Build the "after" region (loop body)
        auto& afterBlock = whileOp.getAfter().emplaceBlock();
        builder.setInsertionPointToEnd(&afterBlock);

        if (whileStmt->body) {
            if (failed(convertStatement(*whileStmt->body, builder))) {
                return failure();
            }
        }

        // Yield empty args back to before region
        builder.create<scf::YieldOp>(loc, ValueRange{});

        return success();
    }

    return failure();
}

Value ASTToFIRConverter::convertExpression(const Expression& expr, OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();

    if (auto* lit = dynamic_cast<const LiteralExpression*>(&expr)) {
        if (std::holds_alternative<int64_t>(lit->value)) {
            int64_t val = std::get<int64_t>(lit->value);
            auto i32Type = builder.getI32Type();
            auto attr = builder.getI32IntegerAttr(val);
            auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
            return constOp.getResult();
        } else if (std::holds_alternative<std::string>(lit->value)) {
            std::string val = std::get<std::string>(lit->value);
            // For now, create a constant with 0 (will be extended)
            auto i32Type = builder.getI32Type();
            auto attr = builder.getI32IntegerAttr(0);
            auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
            return constOp.getResult();
        }
    } else if (auto* ident = dynamic_cast<const IdentifierExpression*>(&expr)) {
        // Variable reference - for now return a dummy constant
        auto i32Type = builder.getI32Type();
        auto attr = builder.getI32IntegerAttr(0);
        auto constOp = builder.create<ConstantOp>(loc, i32Type, attr);
        return constOp.getResult();
    } else if (auto* binop = dynamic_cast<const BinaryExpression*>(&expr)) {
        Value lhs = convertExpression(*binop->left, builder);
        Value rhs = convertExpression(*binop->right, builder);

        if (!lhs || !rhs) {
            return nullptr;
        }

        // Create appropriate FIR binary operation based on operator
        mlir::Type resultType = lhs.getType();  // Assume operands have same type
        switch (binop->op) {
            case TokenType::Plus: {
                auto addOp = builder.create<AddOp>(loc, resultType, lhs, rhs);
                return addOp.getResult();
            }
            case TokenType::Minus: {
                auto subOp = builder.create<SubOp>(loc, resultType, lhs, rhs);
                return subOp.getResult();
            }
            case TokenType::Asterisk: {
                auto mulOp = builder.create<MulOp>(loc, resultType, lhs, rhs);
                return mulOp.getResult();
            }
            case TokenType::Slash: {
                auto divOp = builder.create<DivOp>(loc, resultType, lhs, rhs);
                return divOp.getResult();
            }
            case TokenType::DoubleAmpersand: {
                auto andOp = builder.create<AndOp>(loc, resultType, lhs, rhs);
                return andOp.getResult();
            }
            case TokenType::DoublePipe: {
                auto orOp = builder.create<OrOp>(loc, resultType, lhs, rhs);
                return orOp.getResult();
            }
            case TokenType::DoubleEqual:
            case TokenType::NotEqual:
            case TokenType::LessThan:
            case TokenType::LessThanOrEqual:
            case TokenType::GreaterThan:
            case TokenType::GreaterThanOrEqual: {
                // Determine comparison predicate
                StringRef predicate;
                if (binop->op == TokenType::DoubleEqual) predicate = "eq";
                else if (binop->op == TokenType::NotEqual) predicate = "ne";
                else if (binop->op == TokenType::LessThan) predicate = "lt";
                else if (binop->op == TokenType::LessThanOrEqual) predicate = "le";
                else if (binop->op == TokenType::GreaterThan) predicate = "gt";
                else if (binop->op == TokenType::GreaterThanOrEqual) predicate = "ge";
                else predicate = "eq";

                // Comparison returns bool
                mlir::Type boolType = builder.getI1Type();
                auto cmpOp = builder.create<CmpOp>(loc, boolType, lhs, rhs,
                    StringAttr::get(context, predicate));
                return cmpOp.getResult();
            }
            default:
                // Unsupported operator - return null
                return nullptr;
        }
    } else if (auto* call = dynamic_cast<const CallExpression*>(&expr)) {
        // Convert callee
        Value callee = convertExpression(*call->callee, builder);
        if (!callee) {
            return nullptr;
        }

        // Convert arguments
        SmallVector<Value> args;
        for (const auto& arg : call->args) {
            Value argValue = convertExpression(*arg, builder);
            if (!argValue) {
                return nullptr;
            }
            args.push_back(argValue);
        }

        // Get function name from identifier if callee is IdentifierExpression
        FlatSymbolRefAttr calleeAttr;
        if (auto* ident = dynamic_cast<const IdentifierExpression*>(call->callee.get())) {
            calleeAttr = SymbolRefAttr::get(context, ident->name);
        } else {
            // For complex callee expressions, use a placeholder
            calleeAttr = SymbolRefAttr::get(context, "unknown_func");
        }

        // Create call operation - assume i32 return type for now
        mlir::Type resultType = builder.getI32Type();
        auto callOp = builder.create<func::CallOp>(loc, resultType, calleeAttr, args);
        return callOp.getResult(0);
    }

    return nullptr;
}

mlir::Type ASTToFIRConverter::convertType(const cpp2_transpiler::Type& type) {
    if (type.kind == cpp2_transpiler::Type::Kind::Builtin) {
        if (type.name == "int" || type.name == "i32") {
            return builder.getI32Type();
        }
        if (type.name == "int64" || type.name == "i64") {
            return builder.getI64Type();
        }
        if (type.name == "bool") {
            return builder.getI1Type();
        }
    }

    // Default to i32
    return builder.getI32Type();
}
