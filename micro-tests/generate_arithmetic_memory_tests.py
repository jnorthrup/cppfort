#!/usr/bin/env python3
"""
Generate arithmetic and memory micro tests for C++ decompilation validation.
Category 2: Arithmetic (80 tests)
Category 3: Memory (120 tests)
"""

import os

test_num = 1

def write_test(directory, filename, code, description):
    """Write a test file with header comment"""
    global test_num
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        f.write(f"// {filename}\n")
        f.write(f"// {description}\n")
        f.write(f"// Test #{test_num:03d}\n\n")
        f.write(code)
    test_num += 1
    print(f"Generated: {directory}/{filename}")

# === CATEGORY 2: ARITHMETIC (80 tests) ===

# Integer arithmetic (20 tests)
write_test("arithmetic", "ar001-add-int.cpp", """
int test_add(int a, int b) {
    return a + b;
}

int main() {
    return test_add(5, 3);
}
""", "Integer addition")

write_test("arithmetic", "ar002-sub-int.cpp", """
int test_sub(int a, int b) {
    return a - b;
}

int main() {
    return test_sub(10, 3);
}
""", "Integer subtraction")

write_test("arithmetic", "ar003-mul-int.cpp", """
int test_mul(int a, int b) {
    return a * b;
}

int main() {
    return test_mul(5, 7);
}
""", "Integer multiplication")

write_test("arithmetic", "ar004-div-int.cpp", """
int test_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

int main() {
    return test_div(20, 4);
}
""", "Integer division")

write_test("arithmetic", "ar005-mod-int.cpp", """
int test_mod(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}

int main() {
    return test_mod(17, 5);
}
""", "Integer modulo")

write_test("arithmetic", "ar006-prefix-increment.cpp", """
int test_prefix_inc(int x) {
    return ++x;
}

int main() {
    return test_prefix_inc(5);
}
""", "Prefix increment")

write_test("arithmetic", "ar007-postfix-increment.cpp", """
int test_postfix_inc(int x) {
    int y = x++;
    return x + y;
}

int main() {
    return test_postfix_inc(5);
}
""", "Postfix increment")

write_test("arithmetic", "ar008-prefix-decrement.cpp", """
int test_prefix_dec(int x) {
    return --x;
}

int main() {
    return test_prefix_dec(5);
}
""", "Prefix decrement")

write_test("arithmetic", "ar009-postfix-decrement.cpp", """
int test_postfix_dec(int x) {
    int y = x--;
    return x + y;
}

int main() {
    return test_postfix_dec(5);
}
""", "Postfix decrement")

write_test("arithmetic", "ar010-unary-minus.cpp", """
int test_unary_minus(int x) {
    return -x;
}

int main() {
    return test_unary_minus(5);
}
""", "Unary minus")

write_test("arithmetic", "ar011-unary-plus.cpp", """
int test_unary_plus(int x) {
    return +x;
}

int main() {
    return test_unary_plus(5);
}
""", "Unary plus")

write_test("arithmetic", "ar012-compound-add.cpp", """
int test_compound_add(int x, int y) {
    x += y;
    return x;
}

int main() {
    return test_compound_add(10, 5);
}
""", "Compound addition (+=)")

write_test("arithmetic", "ar013-compound-sub.cpp", """
int test_compound_sub(int x, int y) {
    x -= y;
    return x;
}

int main() {
    return test_compound_sub(10, 3);
}
""", "Compound subtraction (-=)")

write_test("arithmetic", "ar014-compound-mul.cpp", """
int test_compound_mul(int x, int y) {
    x *= y;
    return x;
}

int main() {
    return test_compound_mul(5, 4);
}
""", "Compound multiplication (*=)")

write_test("arithmetic", "ar015-compound-div.cpp", """
int test_compound_div(int x, int y) {
    if (y == 0) return 0;
    x /= y;
    return x;
}

int main() {
    return test_compound_div(20, 4);
}
""", "Compound division (/=)")

write_test("arithmetic", "ar016-compound-mod.cpp", """
int test_compound_mod(int x, int y) {
    if (y == 0) return 0;
    x %= y;
    return x;
}

int main() {
    return test_compound_mod(17, 5);
}
""", "Compound modulo (%=)")

write_test("arithmetic", "ar017-mixed-operations.cpp", """
int test_mixed_ops(int a, int b, int c) {
    return a * b + c;
}

int main() {
    return test_mixed_ops(3, 4, 5);
}
""", "Mixed arithmetic operations")

write_test("arithmetic", "ar018-precedence.cpp", """
int test_precedence(int a, int b, int c) {
    return a + b * c;
}

int main() {
    return test_precedence(2, 3, 4);
}
""", "Operator precedence")

write_test("arithmetic", "ar019-parentheses.cpp", """
int test_parentheses(int a, int b, int c) {
    return (a + b) * c;
}

int main() {
    return test_parentheses(2, 3, 4);
}
""", "Parentheses altering precedence")

write_test("arithmetic", "ar020-complex-expr.cpp", """
int test_complex(int a, int b, int c, int d) {
    return (a + b) * (c - d) / 2;
}

int main() {
    return test_complex(10, 5, 20, 4);
}
""", "Complex arithmetic expression")

# Floating point (15 tests)
write_test("arithmetic", "ar021-add-float.cpp", """
float test_add_float(float a, float b) {
    return a + b;
}

int main() {
    return (int)test_add_float(3.5f, 2.5f);
}
""", "Float addition")

write_test("arithmetic", "ar022-sub-double.cpp", """
double test_sub_double(double a, double b) {
    return a - b;
}

int main() {
    return (int)test_sub_double(10.5, 3.2);
}
""", "Double subtraction")

write_test("arithmetic", "ar023-mul-float.cpp", """
float test_mul_float(float a, float b) {
    return a * b;
}

int main() {
    return (int)test_mul_float(3.0f, 4.0f);
}
""", "Float multiplication")

write_test("arithmetic", "ar024-div-double.cpp", """
double test_div_double(double a, double b) {
    if (b == 0.0) return 0.0;
    return a / b;
}

int main() {
    return (int)test_div_double(20.0, 4.0);
}
""", "Double division")

write_test("arithmetic", "ar025-float-to-int.cpp", """
int test_float_to_int(float x) {
    return (int)x;
}

int main() {
    return test_float_to_int(3.7f);
}
""", "Float to int conversion")

write_test("arithmetic", "ar026-int-to-float.cpp", """
float test_int_to_float(int x) {
    return (float)x;
}

int main() {
    return (int)test_int_to_float(5);
}
""", "Int to float conversion")

write_test("arithmetic", "ar027-float-comparison.cpp", """
int test_float_compare(float a, float b) {
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

int main() {
    return test_float_compare(3.5f, 2.5f);
}
""", "Float comparison")

write_test("arithmetic", "ar028-float-increment.cpp", """
float test_float_inc(float x) {
    return ++x;
}

int main() {
    return (int)test_float_inc(5.5f);
}
""", "Float increment")

write_test("arithmetic", "ar029-double-compound.cpp", """
double test_double_compound(double x, double y) {
    x += y;
    x *= 2.0;
    return x;
}

int main() {
    return (int)test_double_compound(3.0, 2.0);
}
""", "Double compound operations")

write_test("arithmetic", "ar030-float-unary-minus.cpp", """
float test_float_neg(float x) {
    return -x;
}

int main() {
    return (int)test_float_neg(5.0f);
}
""", "Float unary minus")

write_test("arithmetic", "ar031-mixed-float-int.cpp", """
float test_mixed_types(int a, float b) {
    return a + b;
}

int main() {
    return (int)test_mixed_types(5, 3.5f);
}
""", "Mixed int and float arithmetic")

write_test("arithmetic", "ar032-long-double.cpp", """
long double test_long_double(long double x, long double y) {
    return x * y;
}

int main() {
    return (int)test_long_double(3.0L, 4.0L);
}
""", "Long double arithmetic")

write_test("arithmetic", "ar033-float-division-zero.cpp", """
float test_float_div_zero(float x) {
    return x / 0.0f;
}

int main() {
    // Returns infinity
    float result = test_float_div_zero(1.0f);
    return result > 1000.0f ? 1 : 0;
}
""", "Float division by zero (infinity)")

write_test("arithmetic", "ar034-float-nan.cpp", """
float test_nan() {
    return 0.0f / 0.0f;
}

int main() {
    float result = test_nan();
    return (result != result) ? 1 : 0;  // NaN != NaN
}
""", "Float NaN generation")

write_test("arithmetic", "ar035-float-precision.cpp", """
double test_precision() {
    double x = 0.1;
    double y = 0.2;
    double z = 0.3;
    return (x + y == z) ? 1 : 0;
}

int main() {
    return test_precision();
}
""", "Float precision issues")

# Bitwise operations (20 tests)
write_test("arithmetic", "ar036-bitwise-and.cpp", """
int test_bitwise_and(int a, int b) {
    return a & b;
}

int main() {
    return test_bitwise_and(0b1100, 0b1010);
}
""", "Bitwise AND")

write_test("arithmetic", "ar037-bitwise-or.cpp", """
int test_bitwise_or(int a, int b) {
    return a | b;
}

int main() {
    return test_bitwise_or(0b1100, 0b1010);
}
""", "Bitwise OR")

write_test("arithmetic", "ar038-bitwise-xor.cpp", """
int test_bitwise_xor(int a, int b) {
    return a ^ b;
}

int main() {
    return test_bitwise_xor(0b1100, 0b1010);
}
""", "Bitwise XOR")

write_test("arithmetic", "ar039-bitwise-not.cpp", """
int test_bitwise_not(int x) {
    return ~x;
}

int main() {
    return test_bitwise_not(0) & 0xFF;
}
""", "Bitwise NOT")

write_test("arithmetic", "ar040-left-shift.cpp", """
int test_left_shift(int x, int n) {
    return x << n;
}

int main() {
    return test_left_shift(5, 2);
}
""", "Left shift")

write_test("arithmetic", "ar041-right-shift.cpp", """
int test_right_shift(int x, int n) {
    return x >> n;
}

int main() {
    return test_right_shift(20, 2);
}
""", "Right shift")

write_test("arithmetic", "ar042-unsigned-right-shift.cpp", """
unsigned int test_unsigned_shift(unsigned int x, int n) {
    return x >> n;
}

int main() {
    return test_unsigned_shift(0x80000000, 1);
}
""", "Unsigned right shift")

write_test("arithmetic", "ar043-compound-and.cpp", """
int test_compound_and(int x, int y) {
    x &= y;
    return x;
}

int main() {
    return test_compound_and(0b1111, 0b1010);
}
""", "Compound bitwise AND (&=)")

write_test("arithmetic", "ar044-compound-or.cpp", """
int test_compound_or(int x, int y) {
    x |= y;
    return x;
}

int main() {
    return test_compound_or(0b1100, 0b0011);
}
""", "Compound bitwise OR (|=)")

write_test("arithmetic", "ar045-compound-xor.cpp", """
int test_compound_xor(int x, int y) {
    x ^= y;
    return x;
}

int main() {
    return test_compound_xor(0b1111, 0b1010);
}
""", "Compound bitwise XOR (^=)")

write_test("arithmetic", "ar046-compound-left-shift.cpp", """
int test_compound_lshift(int x, int n) {
    x <<= n;
    return x;
}

int main() {
    return test_compound_lshift(5, 2);
}
""", "Compound left shift (<<=)")

write_test("arithmetic", "ar047-compound-right-shift.cpp", """
int test_compound_rshift(int x, int n) {
    x >>= n;
    return x;
}

int main() {
    return test_compound_rshift(20, 2);
}
""", "Compound right shift (>>=)")

write_test("arithmetic", "ar048-bit-mask.cpp", """
int test_bit_mask(int value) {
    return value & 0xFF;
}

int main() {
    return test_bit_mask(0x12345678);
}
""", "Bit masking")

write_test("arithmetic", "ar049-bit-set.cpp", """
int test_bit_set(int value, int bit) {
    return value | (1 << bit);
}

int main() {
    return test_bit_set(0, 3);
}
""", "Set bit")

write_test("arithmetic", "ar050-bit-clear.cpp", """
int test_bit_clear(int value, int bit) {
    return value & ~(1 << bit);
}

int main() {
    return test_bit_clear(0xFF, 3);
}
""", "Clear bit")

write_test("arithmetic", "ar051-bit-toggle.cpp", """
int test_bit_toggle(int value, int bit) {
    return value ^ (1 << bit);
}

int main() {
    return test_bit_toggle(0b1010, 1);
}
""", "Toggle bit")

write_test("arithmetic", "ar052-bit-test.cpp", """
int test_bit_test(int value, int bit) {
    return (value & (1 << bit)) != 0;
}

int main() {
    return test_bit_test(0b1000, 3);
}
""", "Test bit")

write_test("arithmetic", "ar053-swap-xor.cpp", """
void test_swap_xor(int& a, int& b) {
    a ^= b;
    b ^= a;
    a ^= b;
}

int main() {
    int x = 5, y = 10;
    test_swap_xor(x, y);
    return x;
}
""", "Swap using XOR")

write_test("arithmetic", "ar054-bit-count.cpp", """
int test_bit_count(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

int main() {
    return test_bit_count(0b10101010);
}
""", "Count set bits")

write_test("arithmetic", "ar055-power-of-two.cpp", """
bool test_is_power_of_two(unsigned int x) {
    return x && !(x & (x - 1));
}

int main() {
    return test_is_power_of_two(16) ? 1 : 0;
}
""", "Check if power of two")

# Comparison operators (15 tests)
write_test("arithmetic", "ar056-equal.cpp", """
int test_equal(int a, int b) {
    return a == b;
}

int main() {
    return test_equal(5, 5);
}
""", "Equality comparison")

write_test("arithmetic", "ar057-not-equal.cpp", """
int test_not_equal(int a, int b) {
    return a != b;
}

int main() {
    return test_not_equal(5, 3);
}
""", "Inequality comparison")

write_test("arithmetic", "ar058-less-than.cpp", """
int test_less_than(int a, int b) {
    return a < b;
}

int main() {
    return test_less_than(3, 5);
}
""", "Less than comparison")

write_test("arithmetic", "ar059-less-equal.cpp", """
int test_less_equal(int a, int b) {
    return a <= b;
}

int main() {
    return test_less_equal(5, 5);
}
""", "Less than or equal comparison")

write_test("arithmetic", "ar060-greater-than.cpp", """
int test_greater_than(int a, int b) {
    return a > b;
}

int main() {
    return test_greater_than(7, 5);
}
""", "Greater than comparison")

write_test("arithmetic", "ar061-greater-equal.cpp", """
int test_greater_equal(int a, int b) {
    return a >= b;
}

int main() {
    return test_greater_equal(5, 5);
}
""", "Greater than or equal comparison")

write_test("arithmetic", "ar062-three-way-compare.cpp", """
#include <compare>

auto test_three_way(int a, int b) {
    return a <=> b;
}

int main() {
    auto result = test_three_way(5, 3);
    return result > 0 ? 1 : 0;
}
""", "Three-way comparison (C++20)")

write_test("arithmetic", "ar063-unsigned-compare.cpp", """
int test_unsigned_compare(unsigned int a, unsigned int b) {
    return a < b;
}

int main() {
    return test_unsigned_compare(5, 10);
}
""", "Unsigned comparison")

write_test("arithmetic", "ar064-signed-unsigned-compare.cpp", """
int test_mixed_sign_compare(int a, unsigned int b) {
    return a < (int)b;
}

int main() {
    return test_mixed_sign_compare(-5, 10);
}
""", "Mixed signed/unsigned comparison")

write_test("arithmetic", "ar065-chained-compare.cpp", """
int test_chained_compare(int a, int b, int c) {
    return (a < b) && (b < c);
}

int main() {
    return test_chained_compare(1, 5, 10);
}
""", "Chained comparisons")

write_test("arithmetic", "ar066-min-function.cpp", """
int test_min(int a, int b) {
    return (a < b) ? a : b;
}

int main() {
    return test_min(5, 3);
}
""", "Min function using comparison")

write_test("arithmetic", "ar067-max-function.cpp", """
int test_max(int a, int b) {
    return (a > b) ? a : b;
}

int main() {
    return test_max(5, 3);
}
""", "Max function using comparison")

write_test("arithmetic", "ar068-clamp-function.cpp", """
int test_clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

int main() {
    return test_clamp(15, 0, 10);
}
""", "Clamp function")

write_test("arithmetic", "ar069-abs-function.cpp", """
int test_abs(int x) {
    return (x < 0) ? -x : x;
}

int main() {
    return test_abs(-5);
}
""", "Absolute value function")

write_test("arithmetic", "ar070-sign-function.cpp", """
int test_sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

int main() {
    return test_sign(-5);
}
""", "Sign function")

# Logical operators (10 tests)
write_test("arithmetic", "ar071-logical-and.cpp", """
int test_logical_and(bool a, bool b) {
    return a && b;
}

int main() {
    return test_logical_and(true, true);
}
""", "Logical AND")

write_test("arithmetic", "ar072-logical-or.cpp", """
int test_logical_or(bool a, bool b) {
    return a || b;
}

int main() {
    return test_logical_or(false, true);
}
""", "Logical OR")

write_test("arithmetic", "ar073-logical-not.cpp", """
int test_logical_not(bool x) {
    return !x;
}

int main() {
    return test_logical_not(false);
}
""", "Logical NOT")

write_test("arithmetic", "ar074-logical-complex.cpp", """
int test_logical_complex(bool a, bool b, bool c) {
    return (a && b) || (!c);
}

int main() {
    return test_logical_complex(false, true, false);
}
""", "Complex logical expression")

write_test("arithmetic", "ar075-de-morgan.cpp", """
int test_de_morgan(bool a, bool b) {
    return !(a && b) == (!a || !b);
}

int main() {
    return test_de_morgan(true, false);
}
""", "De Morgan's laws")

write_test("arithmetic", "ar076-truthiness.cpp", """
int test_truthiness(int x) {
    return x ? 1 : 0;
}

int main() {
    return test_truthiness(42);
}
""", "Truthiness of integer")

write_test("arithmetic", "ar077-boolean-conversion.cpp", """
bool test_bool_conversion(int x) {
    return static_cast<bool>(x);
}

int main() {
    return test_bool_conversion(0) ? 1 : 0;
}
""", "Boolean conversion")

write_test("arithmetic", "ar078-logical-precedence.cpp", """
int test_logical_precedence(bool a, bool b, bool c) {
    return a || b && c;
}

int main() {
    return test_logical_precedence(false, true, true);
}
""", "Logical operator precedence")

write_test("arithmetic", "ar079-bitwise-vs-logical.cpp", """
int test_bitwise_vs_logical(int a, int b) {
    return (a & b) != (a && b);
}

int main() {
    return test_bitwise_vs_logical(2, 3);
}
""", "Bitwise vs logical operators")

write_test("arithmetic", "ar080-boolean-algebra.cpp", """
bool test_boolean_algebra(bool a, bool b, bool c) {
    return (a && (b || c)) == ((a && b) || (a && c));
}

int main() {
    return test_boolean_algebra(true, false, true);
}
""", "Boolean algebra distributive law")

print(f"\nArithmetic tests complete! Created {test_num-1} tests so far.")
test_num_arithmetic = test_num

# === CATEGORY 3: MEMORY (120 tests) ===

# Stack variables (15 tests)
write_test("memory", "mem001-local-variable.cpp", """
int test_local_variable() {
    int x = 42;
    return x;
}

int main() {
    return test_local_variable();
}
""", "Simple local variable")

write_test("memory", "mem002-multiple-locals.cpp", """
int test_multiple_locals() {
    int a = 1;
    int b = 2;
    int c = 3;
    return a + b + c;
}

int main() {
    return test_multiple_locals();
}
""", "Multiple local variables")

write_test("memory", "mem003-local-array.cpp", """
int test_local_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    return arr[2];
}

int main() {
    return test_local_array();
}
""", "Local array on stack")

write_test("memory", "mem004-static-local.cpp", """
int test_static_local() {
    static int counter = 0;
    return ++counter;
}

int main() {
    test_static_local();
    return test_static_local();
}
""", "Static local variable")

write_test("memory", "mem005-const-local.cpp", """
int test_const_local() {
    const int x = 42;
    return x;
}

int main() {
    return test_const_local();
}
""", "Const local variable")

write_test("memory", "mem006-scope-shadowing.cpp", """
int test_shadowing() {
    int x = 10;
    {
        int x = 20;
        return x;
    }
}

int main() {
    return test_shadowing();
}
""", "Variable shadowing in nested scope")

write_test("memory", "mem007-uninitialized-local.cpp", """
int test_uninitialized() {
    int x;
    x = 42;
    return x;
}

int main() {
    return test_uninitialized();
}
""", "Uninitialized then assigned local")

write_test("memory", "mem008-aggregate-init.cpp", """
struct Point { int x; int y; };

int test_aggregate_init() {
    Point p = {3, 4};
    return p.x + p.y;
}

int main() {
    return test_aggregate_init();
}
""", "Aggregate initialization")

write_test("memory", "mem009-array-decay.cpp", """
int test_array_decay(int arr[]) {
    return arr[0];
}

int main() {
    int arr[3] = {1, 2, 3};
    return test_array_decay(arr);
}
""", "Array decay to pointer")

write_test("memory", "mem010-vla.cpp", """
int test_vla(int n) {
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    return arr[n-1];
}

int main() {
    return test_vla(5);
}
""", "Variable-length array (VLA)")

write_test("memory", "mem011-string-literal.cpp", """
const char* test_string_literal() {
    return "Hello";
}

int main() {
    return test_string_literal()[0];
}
""", "String literal")

write_test("memory", "mem012-global-variable.cpp", """
int global_var = 42;

int test_global() {
    return global_var;
}

int main() {
    return test_global();
}
""", "Global variable access")

write_test("memory", "mem013-global-array.cpp", """
int global_arr[5] = {1, 2, 3, 4, 5};

int test_global_array() {
    return global_arr[2];
}

int main() {
    return test_global_array();
}
""", "Global array access")

write_test("memory", "mem014-static-global.cpp", """
static int static_global = 10;

int test_static_global() {
    return static_global;
}

int main() {
    return test_static_global();
}
""", "Static global variable")

write_test("memory", "mem015-extern-variable.cpp", """
extern int extern_var;
int extern_var = 100;

int test_extern() {
    return extern_var;
}

int main() {
    return test_extern();
}
""", "Extern variable")

# Pointers (30 tests)
write_test("memory", "mem016-pointer-basic.cpp", """
int test_pointer_basic() {
    int x = 42;
    int* ptr = &x;
    return *ptr;
}

int main() {
    return test_pointer_basic();
}
""", "Basic pointer dereference")

write_test("memory", "mem017-pointer-assignment.cpp", """
int test_pointer_assignment() {
    int x = 10;
    int* ptr = &x;
    *ptr = 20;
    return x;
}

int main() {
    return test_pointer_assignment();
}
""", "Assignment through pointer")

write_test("memory", "mem018-null-pointer.cpp", """
int test_null_pointer() {
    int* ptr = nullptr;
    return (ptr == nullptr) ? 1 : 0;
}

int main() {
    return test_null_pointer();
}
""", "Null pointer check")

write_test("memory", "mem019-pointer-arithmetic.cpp", """
int test_pointer_arithmetic() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    ptr += 2;
    return *ptr;
}

int main() {
    return test_pointer_arithmetic();
}
""", "Pointer arithmetic")

write_test("memory", "mem020-pointer-increment.cpp", """
int test_pointer_increment() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return *(++ptr);
}

int main() {
    return test_pointer_increment();
}
""", "Pointer increment")

write_test("memory", "mem021-pointer-decrement.cpp", """
int test_pointer_decrement() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr + 2;
    return *(--ptr);
}

int main() {
    return test_pointer_decrement();
}
""", "Pointer decrement")

write_test("memory", "mem022-pointer-difference.cpp", """
int test_pointer_difference() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr1 = arr;
    int* ptr2 = arr + 3;
    return ptr2 - ptr1;
}

int main() {
    return test_pointer_difference();
}
""", "Pointer difference")

write_test("memory", "mem023-pointer-comparison.cpp", """
int test_pointer_comparison() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr1 = arr;
    int* ptr2 = arr + 3;
    return ptr1 < ptr2;
}

int main() {
    return test_pointer_comparison();
}
""", "Pointer comparison")

write_test("memory", "mem024-double-pointer.cpp", """
int test_double_pointer() {
    int x = 42;
    int* ptr = &x;
    int** pptr = &ptr;
    return **pptr;
}

int main() {
    return test_double_pointer();
}
""", "Double pointer (pointer to pointer)")

write_test("memory", "mem025-pointer-to-array.cpp", """
int test_pointer_to_array() {
    int arr[3] = {1, 2, 3};
    int (*ptr)[3] = &arr;
    return (*ptr)[1];
}

int main() {
    return test_pointer_to_array();
}
""", "Pointer to array")

write_test("memory", "mem026-array-of-pointers.cpp", """
int test_array_of_pointers() {
    int a = 1, b = 2, c = 3;
    int* arr[3] = {&a, &b, &c};
    return *arr[1];
}

int main() {
    return test_array_of_pointers();
}
""", "Array of pointers")

write_test("memory", "mem027-void-pointer.cpp", """
int test_void_pointer() {
    int x = 42;
    void* vptr = &x;
    int* ptr = (int*)vptr;
    return *ptr;
}

int main() {
    return test_void_pointer();
}
""", "Void pointer casting")

write_test("memory", "mem028-const-pointer.cpp", """
int test_const_pointer() {
    int x = 42;
    const int* ptr = &x;
    return *ptr;
}

int main() {
    return test_const_pointer();
}
""", "Pointer to const")

write_test("memory", "mem029-pointer-const.cpp", """
int test_pointer_const() {
    int x = 42, y = 10;
    int* const ptr = &x;
    // ptr = &y;  // Error: can't reassign
    return *ptr;
}

int main() {
    return test_pointer_const();
}
""", "Const pointer")

write_test("memory", "mem030-const-pointer-const.cpp", """
int test_const_pointer_const() {
    int x = 42;
    const int* const ptr = &x;
    return *ptr;
}

int main() {
    return test_const_pointer_const();
}
""", "Const pointer to const")

write_test("memory", "mem031-pointer-iteration.cpp", """
int test_pointer_iteration() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int* ptr = arr; ptr < arr + 5; ptr++) {
        sum += *ptr;
    }
    return sum;
}

int main() {
    return test_pointer_iteration();
}
""", "Array iteration with pointers")

write_test("memory", "mem032-pointer-swap.cpp", """
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(&x, &y);
    return x;
}
""", "Swap using pointers")

write_test("memory", "mem033-function-pointer.cpp", """
int add(int a, int b) { return a + b; }

int test_function_pointer() {
    int (*fptr)(int, int) = add;
    return fptr(3, 4);
}

int main() {
    return test_function_pointer();
}
""", "Function pointer")

write_test("memory", "mem034-pointer-offset.cpp", """
int test_pointer_offset() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return *(ptr + 2);
}

int main() {
    return test_pointer_offset();
}
""", "Pointer offset access")

write_test("memory", "mem035-pointer-array-equivalence.cpp", """
int test_pointer_array_equiv() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return ptr[2] == arr[2];
}

int main() {
    return test_pointer_array_equiv();
}
""", "Pointer/array equivalence")

write_test("memory", "mem036-pointer-to-struct.cpp", """
struct Point { int x; int y; };

int test_pointer_to_struct() {
    Point p = {3, 4};
    Point* ptr = &p;
    return ptr->x + ptr->y;
}

int main() {
    return test_pointer_to_struct();
}
""", "Pointer to struct with arrow operator")

write_test("memory", "mem037-this-pointer.cpp", """
class Counter {
    int value;
public:
    Counter(int v) : value(v) {}
    int getValue() { return this->value; }
};

int main() {
    Counter c(42);
    return c.getValue();
}
""", "This pointer in member function")

write_test("memory", "mem038-pointer-return.cpp", """
int* get_pointer() {
    static int x = 42;
    return &x;
}

int main() {
    int* ptr = get_pointer();
    return *ptr;
}
""", "Return pointer from function")

write_test("memory", "mem039-dangling-pointer.cpp", """
int* dangling_pointer() {
    int x = 42;
    return &x;  // Dangling pointer!
}

int main() {
    // This is undefined behavior, but we'll return a safe value
    return 0;
}
""", "Dangling pointer (undefined behavior)")

write_test("memory", "mem040-pointer-aliasing.cpp", """
int test_pointer_aliasing() {
    int x = 10;
    int* ptr1 = &x;
    int* ptr2 = &x;
    *ptr1 = 20;
    return *ptr2;
}

int main() {
    return test_pointer_aliasing();
}
""", "Pointer aliasing")

write_test("memory", "mem041-restrict-pointer.cpp", """
int test_restrict(int* __restrict a, int* __restrict b) {
    *a = 10;
    *b = 20;
    return *a;
}

int main() {
    int x = 0, y = 0;
    return test_restrict(&x, &y);
}
""", "Restrict pointer qualifier")

write_test("memory", "mem042-pointer-cast.cpp", """
int test_pointer_cast() {
    int x = 0x12345678;
    char* ptr = (char*)&x;
    return ptr[0];
}

int main() {
    return test_pointer_cast();
}
""", "Pointer type casting")

write_test("memory", "mem043-offsetof-pattern.cpp", """
struct Data {
    int a;
    int b;
    int c;
};

int test_offsetof() {
    Data d = {1, 2, 3};
    char* ptr = (char*)&d;
    int* b_ptr = (int*)(ptr + sizeof(int));
    return *b_ptr;
}

int main() {
    return test_offsetof();
}
""", "Manual offset calculation (offsetof pattern)")

write_test("memory", "mem044-pointer-to-member.cpp", """
struct Data {
    int x;
    int y;
};

int test_pointer_to_member() {
    Data d = {3, 4};
    int Data::*ptr = &Data::x;
    return d.*ptr;
}

int main() {
    return test_pointer_to_member();
}
""", "Pointer to member variable")

write_test("memory", "mem045-pointer-to-member-func.cpp", """
struct Calculator {
    int add(int a, int b) { return a + b; }
};

int test_pointer_to_member_func() {
    Calculator calc;
    int (Calculator::*fptr)(int, int) = &Calculator::add;
    return (calc.*fptr)(3, 4);
}

int main() {
    return test_pointer_to_member_func();
}
""", "Pointer to member function")

# Dynamic allocation (25 tests)
write_test("memory", "mem046-new-delete.cpp", """
int test_new_delete() {
    int* ptr = new int(42);
    int result = *ptr;
    delete ptr;
    return result;
}

int main() {
    return test_new_delete();
}
""", "Basic new and delete")

write_test("memory", "mem047-new-array.cpp", """
int test_new_array() {
    int* arr = new int[5]{1, 2, 3, 4, 5};
    int result = arr[2];
    delete[] arr;
    return result;
}

int main() {
    return test_new_array();
}
""", "New and delete array")

write_test("memory", "mem048-new-struct.cpp", """
struct Point { int x; int y; };

int test_new_struct() {
    Point* ptr = new Point{3, 4};
    int result = ptr->x + ptr->y;
    delete ptr;
    return result;
}

int main() {
    return test_new_struct();
}
""", "New and delete struct")

write_test("memory", "mem049-new-class.cpp", """
class Counter {
    int value;
public:
    Counter(int v) : value(v) {}
    int getValue() { return value; }
};

int test_new_class() {
    Counter* ptr = new Counter(42);
    int result = ptr->getValue();
    delete ptr;
    return result;
}

int main() {
    return test_new_class();
}
""", "New and delete class object")

write_test("memory", "mem050-placement-new.cpp", """
#include <new>

int test_placement_new() {
    char buffer[sizeof(int)];
    int* ptr = new (buffer) int(42);
    int result = *ptr;
    ptr->~int();
    return result;
}

int main() {
    return test_placement_new();
}
""", "Placement new")

write_test("memory", "mem051-nothrow-new.cpp", """
#include <new>

int test_nothrow_new() {
    int* ptr = new (std::nothrow) int(42);
    if (!ptr) return -1;
    int result = *ptr;
    delete ptr;
    return result;
}

int main() {
    return test_nothrow_new();
}
""", "Nothrow new")

write_test("memory", "mem052-memory-leak.cpp", """
int test_memory_leak() {
    int* ptr = new int(42);
    // Intentional leak (no delete)
    return *ptr;
}

int main() {
    return test_memory_leak();
}
""", "Memory leak (no delete)")

write_test("memory", "mem053-double-delete.cpp", """
int test_double_delete() {
    int* ptr = new int(42);
    int result = *ptr;
    delete ptr;
    // delete ptr;  // Undefined behavior!
    return result;
}

int main() {
    return test_double_delete();
}
""", "Double delete (undefined behavior)")

write_test("memory", "mem054-delete-null.cpp", """
int test_delete_null() {
    int* ptr = nullptr;
    delete ptr;  // Safe: deleting null is no-op
    return 0;
}

int main() {
    return test_delete_null();
}
""", "Delete null pointer (safe)")

write_test("memory", "mem055-malloc-free.cpp", """
#include <cstdlib>

int test_malloc_free() {
    int* ptr = (int*)malloc(sizeof(int));
    *ptr = 42;
    int result = *ptr;
    free(ptr);
    return result;
}

int main() {
    return test_malloc_free();
}
""", "Malloc and free")

write_test("memory", "mem056-calloc.cpp", """
#include <cstdlib>

int test_calloc() {
    int* arr = (int*)calloc(5, sizeof(int));
    arr[2] = 42;
    int result = arr[2];
    free(arr);
    return result;
}

int main() {
    return test_calloc();
}
""", "Calloc (zero-initialized allocation)")

write_test("memory", "mem057-realloc.cpp", """
#include <cstdlib>

int test_realloc() {
    int* arr = (int*)malloc(5 * sizeof(int));
    arr[2] = 42;
    arr = (int*)realloc(arr, 10 * sizeof(int));
    int result = arr[2];
    free(arr);
    return result;
}

int main() {
    return test_realloc();
}
""", "Realloc")

write_test("memory", "mem058-aligned-alloc.cpp", """
#include <cstdlib>

int test_aligned_alloc() {
    int* ptr = (int*)aligned_alloc(64, sizeof(int));
    if (!ptr) return -1;
    *ptr = 42;
    int result = *ptr;
    free(ptr);
    return result;
}

int main() {
    return test_aligned_alloc();
}
""", "Aligned allocation")

write_test("memory", "mem059-unique-ptr.cpp", """
#include <memory>

int test_unique_ptr() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    return *ptr;
}

int main() {
    return test_unique_ptr();
}
""", "Unique pointer (RAII)")

write_test("memory", "mem060-unique-ptr-array.cpp", """
#include <memory>

int test_unique_ptr_array() {
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    arr[2] = 42;
    return arr[2];
}

int main() {
    return test_unique_ptr_array();
}
""", "Unique pointer to array")

write_test("memory", "mem061-unique-ptr-move.cpp", """
#include <memory>

int test_unique_ptr_move() {
    std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    return *ptr2;
}

int main() {
    return test_unique_ptr_move();
}
""", "Unique pointer move semantics")

write_test("memory", "mem062-shared-ptr.cpp", """
#include <memory>

int test_shared_ptr() {
    std::shared_ptr<int> ptr = std::make_shared<int>(42);
    return *ptr;
}

int main() {
    return test_shared_ptr();
}
""", "Shared pointer")

write_test("memory", "mem063-shared-ptr-copy.cpp", """
#include <memory>

int test_shared_ptr_copy() {
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::shared_ptr<int> ptr2 = ptr1;
    return *ptr2;
}

int main() {
    return test_shared_ptr_copy();
}
""", "Shared pointer copy (reference counting)")

write_test("memory", "mem064-shared-ptr-use-count.cpp", """
#include <memory>

int test_shared_ptr_use_count() {
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::shared_ptr<int> ptr2 = ptr1;
    return ptr1.use_count();
}

int main() {
    return test_shared_ptr_use_count();
}
""", "Shared pointer use count")

write_test("memory", "mem065-weak-ptr.cpp", """
#include <memory>

int test_weak_ptr() {
    std::shared_ptr<int> sptr = std::make_shared<int>(42);
    std::weak_ptr<int> wptr = sptr;
    std::shared_ptr<int> sptr2 = wptr.lock();
    return sptr2 ? *sptr2 : -1;
}

int main() {
    return test_weak_ptr();
}
""", "Weak pointer")

write_test("memory", "mem066-weak-ptr-expired.cpp", """
#include <memory>

int test_weak_ptr_expired() {
    std::weak_ptr<int> wptr;
    {
        std::shared_ptr<int> sptr = std::make_shared<int>(42);
        wptr = sptr;
    }
    return wptr.expired() ? 1 : 0;
}

int main() {
    return test_weak_ptr_expired();
}
""", "Weak pointer expired check")

write_test("memory", "mem067-custom-deleter.cpp", """
#include <memory>

void custom_deleter(int* ptr) {
    delete ptr;
}

int test_custom_deleter() {
    std::shared_ptr<int> ptr(new int(42), custom_deleter);
    return *ptr;
}

int main() {
    return test_custom_deleter();
}
""", "Smart pointer with custom deleter")

write_test("memory", "mem068-make-shared-array.cpp", """
#include <memory>

int test_make_shared_array() {
    std::shared_ptr<int[]> arr = std::make_shared<int[]>(5);
    arr[2] = 42;
    return arr[2];
}

int main() {
    return test_make_shared_array();
}
""", "Make shared for arrays (C++20)")

write_test("memory", "mem069-enable-shared-from-this.cpp", """
#include <memory>

class Node : public std::enable_shared_from_this<Node> {
public:
    int value;
    Node(int v) : value(v) {}
    std::shared_ptr<Node> getPtr() {
        return shared_from_this();
    }
};

int main() {
    std::shared_ptr<Node> node = std::make_shared<Node>(42);
    std::shared_ptr<Node> ptr = node->getPtr();
    return ptr->value;
}
""", "Enable shared from this")

write_test("memory", "mem070-dynamic-2d-array.cpp", """
int test_dynamic_2d_array() {
    int** arr = new int*[3];
    for (int i = 0; i < 3; i++) {
        arr[i] = new int[3];
    }
    arr[1][1] = 42;
    int result = arr[1][1];
    for (int i = 0; i < 3; i++) {
        delete[] arr[i];
    }
    delete[] arr;
    return result;
}

int main() {
    return test_dynamic_2d_array();
}
""", "Dynamic 2D array allocation")

# References (15 tests)
write_test("memory", "mem071-reference-basic.cpp", """
int test_reference() {
    int x = 42;
    int& ref = x;
    return ref;
}

int main() {
    return test_reference();
}
""", "Basic reference")

write_test("memory", "mem072-reference-assignment.cpp", """
int test_reference_assignment() {
    int x = 10;
    int& ref = x;
    ref = 20;
    return x;
}

int main() {
    return test_reference_assignment();
}
""", "Assignment through reference")

write_test("memory", "mem073-const-reference.cpp", """
int test_const_reference() {
    int x = 42;
    const int& ref = x;
    return ref;
}

int main() {
    return test_const_reference();
}
""", "Const reference")

write_test("memory", "mem074-reference-parameter.cpp", """
void increment(int& x) {
    x++;
}

int main() {
    int value = 41;
    increment(value);
    return value;
}
""", "Reference parameter")

write_test("memory", "mem075-const-reference-parameter.cpp", """
int get_value(const int& x) {
    return x;
}

int main() {
    return get_value(42);
}
""", "Const reference parameter")

write_test("memory", "mem076-rvalue-reference.cpp", """
int test_rvalue_reference(int&& x) {
    return x;
}

int main() {
    return test_rvalue_reference(42);
}
""", "Rvalue reference")

write_test("memory", "mem077-move-semantics.cpp", """
#include <utility>

int test_move() {
    int x = 42;
    int y = std::move(x);
    return y;
}

int main() {
    return test_move();
}
""", "Move semantics with std::move")

write_test("memory", "mem078-perfect-forwarding.cpp", """
#include <utility>

int process(int& x) { return x; }

template<typename T>
int forward_call(T&& arg) {
    return process(std::forward<T>(arg));
}

int main() {
    int x = 42;
    return forward_call(x);
}
""", "Perfect forwarding")

write_test("memory", "mem079-reference-return.cpp", """
int& get_reference() {
    static int x = 42;
    return x;
}

int main() {
    int& ref = get_reference();
    return ref;
}
""", "Return reference from function")

write_test("memory", "mem080-dangling-reference.cpp", """
int& dangling_reference() {
    int x = 42;
    return x;  // Dangling reference!
}

int main() {
    // Undefined behavior, return safe value
    return 0;
}
""", "Dangling reference (undefined behavior)")

write_test("memory", "mem081-reference-to-pointer.cpp", """
int test_reference_to_pointer() {
    int x = 42;
    int* ptr = &x;
    int*& ref = ptr;
    return *ref;
}

int main() {
    return test_reference_to_pointer();
}
""", "Reference to pointer")

write_test("memory", "mem082-reference-array.cpp", """
int test_reference_array() {
    int arr[3] = {1, 2, 3};
    int (&ref)[3] = arr;
    return ref[1];
}

int main() {
    return test_reference_array();
}
""", "Reference to array")

write_test("memory", "mem083-reference-swap.cpp", """
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(x, y);
    return x;
}
""", "Swap using references")

write_test("memory", "mem084-reference-chaining.cpp", """
int test_reference_chaining() {
    int x = 42;
    int& ref1 = x;
    int& ref2 = ref1;
    return ref2;
}

int main() {
    return test_reference_chaining();
}
""", "Reference chaining")

write_test("memory", "mem085-reference-vs-pointer.cpp", """
int test_ref_vs_ptr() {
    int x = 42;
    int& ref = x;
    int* ptr = &x;
    return (ref == *ptr) ? 1 : 0;
}

int main() {
    return test_ref_vs_ptr();
}
""", "Reference vs pointer comparison")

# Arrays (20 tests)
write_test("memory", "mem086-fixed-array.cpp", """
int test_fixed_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    return arr[2];
}

int main() {
    return test_fixed_array();
}
""", "Fixed-size array")

write_test("memory", "mem087-array-zero-init.cpp", """
int test_array_zero_init() {
    int arr[5] = {0};
    return arr[0] + arr[4];
}

int main() {
    return test_array_zero_init();
}
""", "Array zero initialization")

write_test("memory", "mem088-array-partial-init.cpp", """
int test_array_partial_init() {
    int arr[5] = {1, 2};
    return arr[0] + arr[4];
}

int main() {
    return test_array_partial_init();
}
""", "Array partial initialization")

write_test("memory", "mem089-multidim-array.cpp", """
int test_multidim_array() {
    int arr[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    return arr[1][1];
}

int main() {
    return test_multidim_array();
}
""", "Multidimensional array")

write_test("memory", "mem090-array-sizeof.cpp", """
int test_array_sizeof() {
    int arr[5] = {1, 2, 3, 4, 5};
    return sizeof(arr) / sizeof(arr[0]);
}

int main() {
    return test_array_sizeof();
}
""", "Array size using sizeof")

write_test("memory", "mem091-array-bounds.cpp", """
int test_array_bounds() {
    int arr[5] = {1, 2, 3, 4, 5};
    // Accessing arr[5] is out of bounds!
    return arr[4];
}

int main() {
    return test_array_bounds();
}
""", "Array bounds (last valid element)")

write_test("memory", "mem092-array-iteration.cpp", """
int test_array_iteration() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_array_iteration();
}
""", "Array iteration")

write_test("memory", "mem093-char-array.cpp", """
int test_char_array() {
    char arr[6] = "Hello";
    return arr[0];
}

int main() {
    return test_char_array();
}
""", "Char array (C-string)")

write_test("memory", "mem094-array-copy.cpp", """
int test_array_copy() {
    int src[3] = {1, 2, 3};
    int dest[3];
    for (int i = 0; i < 3; i++) {
        dest[i] = src[i];
    }
    return dest[1];
}

int main() {
    return test_array_copy();
}
""", "Array copy")

write_test("memory", "mem095-array-comparison.cpp", """
int test_array_comparison() {
    int arr1[3] = {1, 2, 3};
    int arr2[3] = {1, 2, 3};
    for (int i = 0; i < 3; i++) {
        if (arr1[i] != arr2[i]) return 0;
    }
    return 1;
}

int main() {
    return test_array_comparison();
}
""", "Array comparison")

write_test("memory", "mem096-jagged-array.cpp", """
int test_jagged_array() {
    int* arr[3];
    int row0[2] = {1, 2};
    int row1[3] = {3, 4, 5};
    int row2[1] = {6};
    arr[0] = row0;
    arr[1] = row1;
    arr[2] = row2;
    return arr[1][1];
}

int main() {
    return test_jagged_array();
}
""", "Jagged array (array of arrays)")

write_test("memory", "mem097-array-of-structs.cpp", """
struct Point { int x; int y; };

int test_array_of_structs() {
    Point arr[3] = {{1, 2}, {3, 4}, {5, 6}};
    return arr[1].x + arr[1].y;
}

int main() {
    return test_array_of_structs();
}
""", "Array of structs")

write_test("memory", "mem098-array-reverse.cpp", """
int test_array_reverse() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5/2; i++) {
        int temp = arr[i];
        arr[i] = arr[4-i];
        arr[4-i] = temp;
    }
    return arr[0];
}

int main() {
    return test_array_reverse();
}
""", "Array reverse in place")

write_test("memory", "mem099-array-find.cpp", """
int test_array_find(int target) {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

int main() {
    return test_array_find(3);
}
""", "Array linear search")

write_test("memory", "mem100-array-max.cpp", """
int test_array_max() {
    int arr[5] = {3, 7, 2, 9, 1};
    int max = arr[0];
    for (int i = 1; i < 5; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

int main() {
    return test_array_max();
}
""", "Find maximum in array")

write_test("memory", "mem101-array-sum.cpp", """
int test_array_sum() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_array_sum();
}
""", "Array sum")

write_test("memory", "mem102-array-product.cpp", """
int test_array_product() {
    int arr[4] = {2, 3, 4, 5};
    int product = 1;
    for (int i = 0; i < 4; i++) {
        product *= arr[i];
    }
    return product;
}

int main() {
    return test_array_product();
}
""", "Array product")

write_test("memory", "mem103-std-array.cpp", """
#include <array>

int test_std_array() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr[2];
}

int main() {
    return test_std_array();
}
""", "std::array (C++11)")

write_test("memory", "mem104-std-array-at.cpp", """
#include <array>

int test_std_array_at() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr.at(2);
}

int main() {
    return test_std_array_at();
}
""", "std::array with bounds checking")

write_test("memory", "mem105-std-array-size.cpp", """
#include <array>

int test_std_array_size() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr.size();
}

int main() {
    return test_std_array_size();
}
""", "std::array size method")

# Alignment and packing (10 tests)
write_test("memory", "mem106-struct-padding.cpp", """
struct Padded {
    char c;
    int i;
    char c2;
};

int test_struct_padding() {
    return sizeof(Padded);
}

int main() {
    return test_struct_padding();
}
""", "Struct padding")

write_test("memory", "mem107-packed-struct.cpp", """
struct __attribute__((packed)) Packed {
    char c;
    int i;
    char c2;
};

int test_packed_struct() {
    return sizeof(Packed);
}

int main() {
    return test_packed_struct();
}
""", "Packed struct (no padding)")

write_test("memory", "mem108-alignas.cpp", """
struct alignas(16) Aligned {
    int x;
};

int test_alignas() {
    return alignof(Aligned);
}

int main() {
    return test_alignas();
}
""", "Alignas specifier")

write_test("memory", "mem109-alignof.cpp", """
int test_alignof() {
    return alignof(int);
}

int main() {
    return test_alignof();
}
""", "Alignof operator")

write_test("memory", "mem110-cache-line-alignment.cpp", """
struct alignas(64) CacheLine {
    int data[16];
};

int test_cache_line() {
    CacheLine cl;
    cl.data[0] = 42;
    return cl.data[0];
}

int main() {
    return test_cache_line();
}
""", "Cache line alignment")

write_test("memory", "mem111-union-type-punning.cpp", """
union Converter {
    int i;
    float f;
};

int test_union_punning() {
    Converter c;
    c.f = 3.14f;
    return c.i != 0;
}

int main() {
    return test_union_punning();
}
""", "Union type punning")

write_test("memory", "mem112-bit-field.cpp", """
struct BitField {
    unsigned int a : 3;
    unsigned int b : 5;
    unsigned int c : 8;
};

int test_bit_field() {
    BitField bf;
    bf.a = 7;
    bf.b = 31;
    bf.c = 255;
    return bf.a + bf.b;
}

int main() {
    return test_bit_field();
}
""", "Bit fields")

write_test("memory", "mem113-zero-size-array.cpp", """
struct FlexibleArray {
    int count;
    int data[];
};

int test_flexible_array() {
    return sizeof(FlexibleArray);
}

int main() {
    return test_flexible_array();
}
""", "Flexible array member (zero-size)")

write_test("memory", "mem114-struct-layout.cpp", """
struct Layout {
    char a;
    short b;
    int c;
    long long d;
};

int test_struct_layout() {
    return sizeof(Layout);
}

int main() {
    return test_struct_layout();
}
""", "Struct memory layout")

write_test("memory", "mem115-anonymous-union.cpp", """
struct Data {
    union {
        int i;
        float f;
    };
};

int test_anonymous_union() {
    Data d;
    d.i = 42;
    return d.i;
}

int main() {
    return test_anonymous_union();
}
""", "Anonymous union in struct")

# Volatile and memory ordering (5 tests)
write_test("memory", "mem116-volatile-read.cpp", """
volatile int global_volatile = 42;

int test_volatile_read() {
    return global_volatile;
}

int main() {
    return test_volatile_read();
}
""", "Volatile variable read")

write_test("memory", "mem117-volatile-write.cpp", """
volatile int global_volatile = 0;

void test_volatile_write(int value) {
    global_volatile = value;
}

int main() {
    test_volatile_write(42);
    return global_volatile;
}
""", "Volatile variable write")

write_test("memory", "mem118-volatile-pointer.cpp", """
int test_volatile_pointer() {
    volatile int x = 42;
    volatile int* ptr = &x;
    return *ptr;
}

int main() {
    return test_volatile_pointer();
}
""", "Volatile pointer")

write_test("memory", "mem119-memory-mapped-io.cpp", """
#define HARDWARE_REG ((volatile unsigned int*)0x40000000)

int test_mmio() {
    // Simulated memory-mapped I/O
    // *HARDWARE_REG = 0x12345678;
    // return *HARDWARE_REG;
    return 42;  // Placeholder
}

int main() {
    return test_mmio();
}
""", "Memory-mapped I/O pattern")

write_test("memory", "mem120-atomic-flag.cpp", """
#include <atomic>

std::atomic_flag flag = ATOMIC_FLAG_INIT;

int test_atomic_flag() {
    bool was_set = flag.test_and_set();
    flag.clear();
    return was_set ? 1 : 0;
}

int main() {
    return test_atomic_flag();
}
""", "Atomic flag operations")

print(f"\nGeneration complete! Created {test_num-1} tests total.")
print(f"Arithmetic tests: {test_num_arithmetic-1}")
print(f"Memory tests: {test_num-test_num_arithmetic}")
