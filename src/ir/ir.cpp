#include "ir.h"
#include <sstream>

namespace ir {

// IRBuilder implementation

TypePtr IRBuilder::makeVoidType() {
    auto type = std::make_shared<Type>();
    type->kind = TypeKind::Void;
    return type;
}

TypePtr IRBuilder::makeIntType(int bits, bool is_signed) {
    std::stringstream key;
    key << (is_signed ? "int" : "uint") << bits;

    auto it = type_cache.find(key.str());
    if (it != type_cache.end()) {
        return it->second;
    }

    auto type = std::make_shared<Type>();
    switch (bits) {
        case 8:  type->kind = is_signed ? TypeKind::Int8 : TypeKind::UInt8; break;
        case 16: type->kind = is_signed ? TypeKind::Int16 : TypeKind::UInt16; break;
        case 32: type->kind = is_signed ? TypeKind::Int32 : TypeKind::UInt32; break;
        case 64: type->kind = is_signed ? TypeKind::Int64 : TypeKind::UInt64; break;
        default: return nullptr;
    }

    type_cache[key.str()] = type;
    return type;
}

TypePtr IRBuilder::makeFloatType(int bits) {
    std::stringstream key;
    key << "float" << bits;

    auto it = type_cache.find(key.str());
    if (it != type_cache.end()) {
        return it->second;
    }

    auto type = std::make_shared<Type>();
    type->kind = (bits == 32) ? TypeKind::Float32 : TypeKind::Float64;

    type_cache[key.str()] = type;
    return type;
}

TypePtr IRBuilder::makePointerType(TypePtr pointee) {
    auto type = std::make_shared<Type>();
    type->kind = TypeKind::Pointer;
    type->pointee = pointee;
    return type;
}

TypePtr IRBuilder::makeArrayType(TypePtr element, size_t size) {
    auto type = std::make_shared<Type>();
    type->kind = TypeKind::Array;
    type->pointee = element;
    type->array_size = size;
    return type;
}

TypePtr IRBuilder::makeFunctionType(TypePtr ret, std::vector<TypePtr> params) {
    auto type = std::make_shared<Type>();
    type->kind = TypeKind::Function;
    type->return_type = ret;
    type->params = std::move(params);
    return type;
}

ExprPtr IRBuilder::makeIntLiteral(int64_t val, TypePtr type) {
    auto expr = std::make_shared<Expression>();
    expr->kind = ExprKind::IntLiteral;
    expr->type = type;
    expr->data = val;
    return expr;
}

ExprPtr IRBuilder::makeStringLiteral(const std::string& str) {
    auto expr = std::make_shared<Expression>();
    expr->kind = ExprKind::StringLiteral;
    expr->type = makePointerType(makeIntType(8, false));  // char*
    expr->data = str;
    return expr;
}

ExprPtr IRBuilder::makeIdentifier(const std::string& name, TypePtr type) {
    auto expr = std::make_shared<Expression>();
    expr->kind = ExprKind::Identifier;
    expr->type = type;
    expr->data = name;
    return expr;
}

ExprPtr IRBuilder::makeBinaryOp(const std::string& op, ExprPtr left, ExprPtr right) {
    auto expr = std::make_shared<Expression>();
    expr->kind = ExprKind::BinaryOp;
    expr->op = op;
    expr->data = std::make_pair(left, right);

    // Type inference for common operators
    if (op == "+" || op == "-" || op == "*" || op == "/" || op == "%") {
        expr->type = left->type;  // Assume same type for arithmetic
    } else if (op == "==" || op == "!=" || op == "<" || op == ">" || op == "<=" || op == ">=") {
        expr->type = std::make_shared<Type>();
        expr->type->kind = TypeKind::Bool;
    } else {
        expr->type = left->type;
    }

    return expr;
}

ExprPtr IRBuilder::makeCall(ExprPtr func, std::vector<ExprPtr> args) {
    auto expr = std::make_shared<Expression>();
    expr->kind = ExprKind::Call;
    expr->data = std::move(args);

    // Extract return type from function type
    if (func->type && func->type->kind == TypeKind::Function) {
        expr->type = func->type->return_type;
    }

    return expr;
}

StmtPtr IRBuilder::makeExprStmt(ExprPtr expr) {
    auto stmt = std::make_shared<Statement>();
    stmt->kind = StmtKind::Expression;
    stmt->condition = expr;
    return stmt;
}

StmtPtr IRBuilder::makeBlock(std::vector<StmtPtr> stmts) {
    auto stmt = std::make_shared<Statement>();
    stmt->kind = StmtKind::Block;
    stmt->body = std::move(stmts);
    return stmt;
}

StmtPtr IRBuilder::makeIf(ExprPtr cond, StmtPtr then_stmt, StmtPtr else_stmt) {
    auto stmt = std::make_shared<Statement>();
    stmt->kind = StmtKind::If;
    stmt->condition = cond;
    stmt->then_stmt = then_stmt;
    stmt->else_stmt = else_stmt;
    return stmt;
}

StmtPtr IRBuilder::makeWhile(ExprPtr cond, StmtPtr body) {
    auto stmt = std::make_shared<Statement>();
    stmt->kind = StmtKind::While;
    stmt->condition = cond;
    stmt->then_stmt = body;
    return stmt;
}

StmtPtr IRBuilder::makeReturn(ExprPtr value) {
    auto stmt = std::make_shared<Statement>();
    stmt->kind = StmtKind::Return;
    stmt->return_value = value;
    return stmt;
}

DeclPtr IRBuilder::makeVariable(const std::string& name, TypePtr type, ExprPtr init) {
    auto decl = std::make_shared<Declaration>();
    decl->kind = DeclKind::Variable;
    decl->name = name;
    decl->type = type;
    decl->initializer = init;
    return decl;
}

FuncPtr IRBuilder::makeFunction(const std::string& name, TypePtr ret_type,
                               std::vector<std::pair<std::string, TypePtr>> params,
                               std::vector<StmtPtr> body) {
    auto func = std::make_shared<Function>();
    func->name = name;
    func->return_type = ret_type;
    func->params = std::move(params);
    func->body = std::move(body);
    return func;
}

std::shared_ptr<Module> IRBuilder::makeModule(const std::string& filename, SourceLang lang) {
    auto module = std::make_shared<Module>();
    module->filename = filename;
    module->source = lang;
    return module;
}

// SemanticAnalyzer implementation

bool SemanticAnalyzer::analyze(const Module& module) {
    errors.clear();

    // Check all functions
    for (const auto& func : module.functions) {
        if (!checkFunction(*func)) {
            return false;
        }
    }

    // Check all declarations
    for (const auto& decl : module.declarations) {
        if (decl->initializer && !checkExpression(*decl->initializer)) {
            return false;
        }
    }

    return errors.empty();
}

bool SemanticAnalyzer::checkType(const Type& type) {
    switch (type.kind) {
        case TypeKind::Pointer:
        case TypeKind::Reference:
            if (!type.pointee) {
                errors.push_back("Pointer/Reference type missing target type");
                return false;
            }
            return checkType(*type.pointee);

        case TypeKind::Array:
            if (!type.pointee) {
                errors.push_back("Array type missing element type");
                return false;
            }
            if (type.array_size == 0) {
                errors.push_back("Array size cannot be zero");
                return false;
            }
            return checkType(*type.pointee);

        case TypeKind::Function:
            if (!type.return_type) {
                errors.push_back("Function type missing return type");
                return false;
            }
            if (!checkType(*type.return_type)) {
                return false;
            }
            for (const auto& param : type.params) {
                if (!checkType(*param)) {
                    return false;
                }
            }
            return true;

        default:
            return true;
    }
}

bool SemanticAnalyzer::checkExpression(const Expression& expr) {
    if (!expr.type) {
        errors.push_back("Expression missing type");
        return false;
    }

    switch (expr.kind) {
        case ExprKind::BinaryOp: {
            auto& operands = std::get<std::pair<ExprPtr, ExprPtr>>(expr.data);
            return checkExpression(*operands.first) &&
                   checkExpression(*operands.second);
        }

        case ExprKind::UnaryOp: {
            // Assuming UnaryOp stores operand in first element of pair
            if (std::holds_alternative<std::pair<ExprPtr, ExprPtr>>(expr.data)) {
                auto& operand = std::get<std::pair<ExprPtr, ExprPtr>>(expr.data).first;
                return checkExpression(*operand);
            }
            return true;
        }

        case ExprKind::Call: {
            auto& args = std::get<std::vector<ExprPtr>>(expr.data);
            for (const auto& arg : args) {
                if (!checkExpression(*arg)) {
                    return false;
                }
            }
            return true;
        }

        default:
            return true;
    }
}

bool SemanticAnalyzer::checkStatement(const Statement& stmt) {
    switch (stmt.kind) {
        case StmtKind::Expression:
            return stmt.condition ? checkExpression(*stmt.condition) : true;

        case StmtKind::If:
            if (!stmt.condition || !checkExpression(*stmt.condition)) {
                errors.push_back("If statement missing or invalid condition");
                return false;
            }
            if (stmt.then_stmt && !checkStatement(*stmt.then_stmt)) {
                return false;
            }
            if (stmt.else_stmt && !checkStatement(*stmt.else_stmt)) {
                return false;
            }
            return true;

        case StmtKind::While:
        case StmtKind::DoWhile:
            if (!stmt.condition || !checkExpression(*stmt.condition)) {
                errors.push_back("Loop missing or invalid condition");
                return false;
            }
            return stmt.then_stmt ? checkStatement(*stmt.then_stmt) : true;

        case StmtKind::Block:
            for (const auto& s : stmt.body) {
                if (!checkStatement(*s)) {
                    return false;
                }
            }
            return true;

        case StmtKind::Return:
            return stmt.return_value ? checkExpression(*stmt.return_value) : true;

        default:
            return true;
    }
}

bool SemanticAnalyzer::checkFunction(const Function& func) {
    if (func.name.empty()) {
        errors.push_back("Function missing name");
        return false;
    }

    if (!func.return_type) {
        errors.push_back("Function missing return type");
        return false;
    }

    if (!checkType(*func.return_type)) {
        return false;
    }

    // Check parameter types
    for (const auto& [param_name, param_type] : func.params) {
        if (!param_type || !checkType(*param_type)) {
            errors.push_back("Invalid parameter type in function " + func.name);
            return false;
        }
    }

    // Check function body
    for (const auto& stmt : func.body) {
        if (!checkStatement(*stmt)) {
            return false;
        }
    }

    return true;
}

} // namespace ir