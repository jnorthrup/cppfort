// Out-of-line destructors for types that hold unique_ptrs to incomplete types.
#include "../include/ast.hpp"

namespace cpp2_transpiler {

Type::~Type() = default;
LambdaExpression::~LambdaExpression() = default;

} // namespace cpp2_transpiler
