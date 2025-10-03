#!/usr/bin/env python3
"""
Generate comprehensive control flow micro tests for C++ decompilation validation.
Category 1: Control Flow (100 tests)
"""

import os

OUTPUT_DIR = "control-flow"

# Test counter
test_num = 1

def write_test(filename, code, description):
    """Write a test file with header comment"""
    global test_num
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(f"// {filename}\n")
        f.write(f"// {description}\n")
        f.write(f"// Test #{test_num:03d}\n\n")
        f.write(code)
    test_num += 1
    print(f"Generated: {filename}")

# === IF STATEMENTS (20 tests) ===

# Simple if
write_test("cf001-simple-if.cpp", """
int test_simple_if(int x) {
    if (x > 0) {
        return 1;
    }
    return 0;
}

int main() {
    return test_simple_if(5);
}
""", "Simple if statement with comparison")

# If-else
write_test("cf002-if-else.cpp", """
int test_if_else(int x) {
    if (x > 0) {
        return 1;
    } else {
        return -1;
    }
}

int main() {
    return test_if_else(5);
}
""", "If-else statement")

# Nested if
write_test("cf003-nested-if.cpp", """
int test_nested_if(int x, int y) {
    if (x > 0) {
        if (y > 0) {
            return 1;
        }
        return 2;
    }
    return 3;
}

int main() {
    return test_nested_if(5, 3);
}
""", "Nested if statements")

# If-else-if chain
write_test("cf004-if-else-if.cpp", """
int test_if_else_if(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else if (x < 10) {
        return 1;
    } else {
        return 2;
    }
}

int main() {
    return test_if_else_if(5);
}
""", "If-else-if chain")

# Complex condition
write_test("cf005-complex-condition.cpp", """
int test_complex_condition(int x, int y) {
    if (x > 0 && y < 10) {
        return 1;
    }
    return 0;
}

int main() {
    return test_complex_condition(5, 3);
}
""", "If with complex logical AND condition")

# OR condition
write_test("cf006-or-condition.cpp", """
int test_or_condition(int x, int y) {
    if (x > 10 || y > 10) {
        return 1;
    }
    return 0;
}

int main() {
    return test_or_condition(5, 15);
}
""", "If with logical OR condition")

# NOT condition
write_test("cf007-not-condition.cpp", """
int test_not_condition(bool flag) {
    if (!flag) {
        return 1;
    }
    return 0;
}

int main() {
    return test_not_condition(false);
}
""", "If with logical NOT condition")

# Multiple statements in if
write_test("cf008-multi-statement-if.cpp", """
int test_multi_statement(int x) {
    int result = 0;
    if (x > 0) {
        result = x * 2;
        result += 10;
        return result;
    }
    return -1;
}

int main() {
    return test_multi_statement(5);
}
""", "If block with multiple statements")

# If without else, side effects
write_test("cf009-if-side-effect.cpp", """
void test_if_side_effect(int x, int& result) {
    if (x > 0) {
        result = x * 2;
    }
}

int main() {
    int r = 0;
    test_if_side_effect(5, r);
    return r;
}
""", "If statement with side effects on reference parameter")

# Early return pattern
write_test("cf010-early-return.cpp", """
int test_early_return(int x) {
    if (x < 0) {
        return 0;
    }
    if (x > 100) {
        return 100;
    }
    return x;
}

int main() {
    return test_early_return(50);
}
""", "Multiple early return guards")

# === TERNARY OPERATOR (5 tests) ===

write_test("cf011-simple-ternary.cpp", """
int test_ternary(int x) {
    return (x > 0) ? 1 : -1;
}

int main() {
    return test_ternary(5);
}
""", "Simple ternary operator")

write_test("cf012-nested-ternary.cpp", """
int test_nested_ternary(int x) {
    return (x > 0) ? (x > 10 ? 2 : 1) : 0;
}

int main() {
    return test_nested_ternary(5);
}
""", "Nested ternary operators")

write_test("cf013-ternary-assignment.cpp", """
int test_ternary_assignment(int x) {
    int result = (x > 0) ? x * 2 : x / 2;
    return result;
}

int main() {
    return test_ternary_assignment(10);
}
""", "Ternary operator in assignment")

write_test("cf014-ternary-function-calls.cpp", """
int positive_func() { return 1; }
int negative_func() { return -1; }

int test_ternary_calls(int x) {
    return (x > 0) ? positive_func() : negative_func();
}

int main() {
    return test_ternary_calls(5);
}
""", "Ternary operator with function calls")

write_test("cf015-ternary-complex-expr.cpp", """
int test_ternary_complex(int x, int y) {
    return (x > y) ? (x + y) : (x - y);
}

int main() {
    return test_ternary_complex(10, 3);
}
""", "Ternary with complex expressions in both branches")

# === FOR LOOPS (15 tests) ===

write_test("cf016-simple-for.cpp", """
int test_simple_for() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_simple_for();
}
""", "Simple for loop with increment")

write_test("cf017-for-decrement.cpp", """
int test_for_decrement() {
    int sum = 0;
    for (int i = 10; i > 0; i--) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_decrement();
}
""", "For loop with decrement")

write_test("cf018-for-step-2.cpp", """
int test_for_step() {
    int sum = 0;
    for (int i = 0; i < 10; i += 2) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_step();
}
""", "For loop with step of 2")

write_test("cf019-nested-for.cpp", """
int test_nested_for() {
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            sum += i * j;
        }
    }
    return sum;
}

int main() {
    return test_nested_for();
}
""", "Nested for loops")

write_test("cf020-for-array-iteration.cpp", """
int test_for_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_for_array();
}
""", "For loop iterating over array")

write_test("cf021-for-break.cpp", """
int test_for_break() {
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        if (i >= 10) break;
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_break();
}
""", "For loop with break statement")

write_test("cf022-for-continue.cpp", """
int test_for_continue() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_continue();
}
""", "For loop with continue statement")

write_test("cf023-for-empty-body.cpp", """
int test_for_empty() {
    int i;
    for (i = 0; i < 10; i++);
    return i;
}

int main() {
    return test_for_empty();
}
""", "For loop with empty body")

write_test("cf024-for-multiple-init.cpp", """
int test_for_multi_init() {
    int sum = 0;
    for (int i = 0, j = 10; i < j; i++, j--) {
        sum += i + j;
    }
    return sum;
}

int main() {
    return test_for_multi_init();
}
""", "For loop with multiple initializers and increments")

write_test("cf025-for-complex-condition.cpp", """
int test_for_complex_cond() {
    int sum = 0;
    for (int i = 0; i < 10 && sum < 20; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_complex_cond();
}
""", "For loop with complex condition")

write_test("cf026-range-for-array.cpp", """
int test_range_for() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}

int main() {
    return test_range_for();
}
""", "Range-based for loop over array")

write_test("cf027-range-for-const-ref.cpp", """
int test_range_for_const_ref() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (const int& x : arr) {
        sum += x;
    }
    return sum;
}

int main() {
    return test_range_for_const_ref();
}
""", "Range-based for with const reference")

write_test("cf028-range-for-modification.cpp", """
void test_range_for_modify(int arr[5]) {
    for (int& x : arr) {
        x *= 2;
    }
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    test_range_for_modify(arr);
    return arr[0] + arr[4];
}
""", "Range-based for with modification via reference")

write_test("cf029-for-infinite-with-break.cpp", """
int test_infinite_for() {
    int sum = 0;
    for (;;) {
        sum++;
        if (sum >= 10) break;
    }
    return sum;
}

int main() {
    return test_infinite_for();
}
""", "Infinite for loop with break")

write_test("cf030-for-backward.cpp", """
int test_for_backward() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 4; i >= 0; i--) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_for_backward();
}
""", "For loop iterating backward")

# === WHILE LOOPS (10 tests) ===

write_test("cf031-simple-while.cpp", """
int test_simple_while() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_simple_while();
}
""", "Simple while loop")

write_test("cf032-while-break.cpp", """
int test_while_break() {
    int sum = 0;
    int i = 0;
    while (true) {
        if (i >= 10) break;
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_while_break();
}
""", "While loop with break")

write_test("cf033-while-continue.cpp", """
int test_while_continue() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        i++;
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_while_continue();
}
""", "While loop with continue")

write_test("cf034-while-complex-condition.cpp", """
int test_while_complex() {
    int sum = 0;
    int i = 0;
    while (i < 10 && sum < 20) {
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_while_complex();
}
""", "While loop with complex condition")

write_test("cf035-nested-while.cpp", """
int test_nested_while() {
    int sum = 0;
    int i = 0;
    while (i < 5) {
        int j = 0;
        while (j < 5) {
            sum += i * j;
            j++;
        }
        i++;
    }
    return sum;
}

int main() {
    return test_nested_while();
}
""", "Nested while loops")

write_test("cf036-while-decrement.cpp", """
int test_while_decrement() {
    int sum = 0;
    int i = 10;
    while (i > 0) {
        sum += i;
        i--;
    }
    return sum;
}

int main() {
    return test_while_decrement();
}
""", "While loop with decrement")

write_test("cf037-while-pointer.cpp", """
int test_while_pointer() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    int sum = 0;
    while (ptr < arr + 5) {
        sum += *ptr;
        ptr++;
    }
    return sum;
}

int main() {
    return test_while_pointer();
}
""", "While loop with pointer iteration")

write_test("cf038-while-assignment.cpp", """
int get_value(int& counter) {
    return counter++;
}

int test_while_assignment() {
    int counter = 0;
    int sum = 0;
    int val;
    while ((val = get_value(counter)) < 10) {
        sum += val;
    }
    return sum;
}

int main() {
    return test_while_assignment();
}
""", "While loop with assignment in condition")

write_test("cf039-while-empty-body.cpp", """
int test_while_empty() {
    int i = 0;
    while (i++ < 10);
    return i;
}

int main() {
    return test_while_empty();
}
""", "While loop with empty body")

write_test("cf040-while-flag-pattern.cpp", """
int test_while_flag() {
    bool found = false;
    int i = 0;
    int result = -1;
    while (!found && i < 10) {
        if (i == 5) {
            found = true;
            result = i;
        }
        i++;
    }
    return result;
}

int main() {
    return test_while_flag();
}
""", "While loop with boolean flag pattern")

# === DO-WHILE LOOPS (8 tests) ===

write_test("cf041-simple-do-while.cpp", """
int test_do_while() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
    } while (i < 10);
    return sum;
}

int main() {
    return test_do_while();
}
""", "Simple do-while loop")

write_test("cf042-do-while-execute-once.cpp", """
int test_do_while_once() {
    int sum = 0;
    do {
        sum = 42;
    } while (false);
    return sum;
}

int main() {
    return test_do_while_once();
}
""", "Do-while that executes exactly once")

write_test("cf043-do-while-break.cpp", """
int test_do_while_break() {
    int sum = 0;
    int i = 0;
    do {
        if (i >= 10) break;
        sum += i;
        i++;
    } while (true);
    return sum;
}

int main() {
    return test_do_while_break();
}
""", "Do-while with break")

write_test("cf044-do-while-continue.cpp", """
int test_do_while_continue() {
    int sum = 0;
    int i = 0;
    do {
        i++;
        if (i % 2 == 0) continue;
        sum += i;
    } while (i < 10);
    return sum;
}

int main() {
    return test_do_while_continue();
}
""", "Do-while with continue")

write_test("cf045-nested-do-while.cpp", """
int test_nested_do_while() {
    int sum = 0;
    int i = 0;
    do {
        int j = 0;
        do {
            sum += i * j;
            j++;
        } while (j < 5);
        i++;
    } while (i < 5);
    return sum;
}

int main() {
    return test_nested_do_while();
}
""", "Nested do-while loops")

write_test("cf046-do-while-complex-condition.cpp", """
int test_do_while_complex() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
    } while (i < 10 && sum < 20);
    return sum;
}

int main() {
    return test_do_while_complex();
}
""", "Do-while with complex condition")

write_test("cf047-do-while-decrement.cpp", """
int test_do_while_decrement() {
    int sum = 0;
    int i = 10;
    do {
        sum += i;
        i--;
    } while (i > 0);
    return sum;
}

int main() {
    return test_do_while_decrement();
}
""", "Do-while with decrement")

write_test("cf048-do-while-assignment.cpp", """
int get_value(int& counter) {
    return counter++;
}

int test_do_while_assignment() {
    int counter = 0;
    int sum = 0;
    int val;
    do {
        val = get_value(counter);
        sum += val;
    } while (val < 10);
    return sum;
}

int main() {
    return test_do_while_assignment();
}
""", "Do-while with assignment")

# === SWITCH STATEMENTS (12 tests) ===

write_test("cf049-simple-switch.cpp", """
int test_simple_switch(int x) {
    switch (x) {
        case 1:
            return 10;
        case 2:
            return 20;
        case 3:
            return 30;
        default:
            return 0;
    }
}

int main() {
    return test_simple_switch(2);
}
""", "Simple switch statement")

write_test("cf050-switch-fallthrough.cpp", """
int test_switch_fallthrough(int x) {
    int result = 0;
    switch (x) {
        case 1:
            result += 1;
        case 2:
            result += 2;
        case 3:
            result += 3;
            break;
        default:
            result = -1;
    }
    return result;
}

int main() {
    return test_switch_fallthrough(1);
}
""", "Switch with fall-through cases")

write_test("cf051-switch-no-default.cpp", """
int test_switch_no_default(int x) {
    int result = -1;
    switch (x) {
        case 1:
            result = 10;
            break;
        case 2:
            result = 20;
            break;
    }
    return result;
}

int main() {
    return test_switch_no_default(1);
}
""", "Switch without default case")

write_test("cf052-switch-multiple-statements.cpp", """
int test_switch_multi_stmt(int x) {
    int result = 0;
    switch (x) {
        case 1:
            result = 5;
            result *= 2;
            result += 3;
            break;
        case 2:
            result = 10;
            result /= 2;
            break;
        default:
            result = 0;
    }
    return result;
}

int main() {
    return test_switch_multi_stmt(1);
}
""", "Switch cases with multiple statements")

write_test("cf053-switch-char.cpp", """
int test_switch_char(char c) {
    switch (c) {
        case 'a':
            return 1;
        case 'b':
            return 2;
        case 'c':
            return 3;
        default:
            return 0;
    }
}

int main() {
    return test_switch_char('b');
}
""", "Switch on char type")

write_test("cf054-switch-enum.cpp", """
enum Color { RED = 0, GREEN = 1, BLUE = 2 };

int test_switch_enum(Color c) {
    switch (c) {
        case RED:
            return 10;
        case GREEN:
            return 20;
        case BLUE:
            return 30;
    }
    return 0;
}

int main() {
    return test_switch_enum(GREEN);
}
""", "Switch on enum type")

write_test("cf055-switch-nested.cpp", """
int test_nested_switch(int x, int y) {
    switch (x) {
        case 1:
            switch (y) {
                case 1: return 11;
                case 2: return 12;
                default: return 10;
            }
        case 2:
            return 20;
        default:
            return 0;
    }
}

int main() {
    return test_nested_switch(1, 2);
}
""", "Nested switch statements")

write_test("cf056-switch-range.cpp", """
int test_switch_range(int x) {
    switch (x) {
        case 0 ... 9:
            return 1;
        case 10 ... 19:
            return 2;
        default:
            return 0;
    }
}

int main() {
    return test_switch_range(15);
}
""", "Switch with GCC range extension")

write_test("cf057-switch-duffs-device.cpp", """
void test_duffs_device(int* dest, const int* src, int count) {
    int n = (count + 7) / 8;
    switch (count % 8) {
        case 0: do { *dest++ = *src++;
        case 7:      *dest++ = *src++;
        case 6:      *dest++ = *src++;
        case 5:      *dest++ = *src++;
        case 4:      *dest++ = *src++;
        case 3:      *dest++ = *src++;
        case 2:      *dest++ = *src++;
        case 1:      *dest++ = *src++;
        } while (--n > 0);
    }
}

int main() {
    int src[10] = {0,1,2,3,4,5,6,7,8,9};
    int dest[10] = {0};
    test_duffs_device(dest, src, 10);
    return dest[5];
}
""", "Duff's device (switch/loop interleaving)")

write_test("cf058-switch-return-in-case.cpp", """
int test_switch_return(int x) {
    switch (x) {
        case 1:
            return 10;
        case 2:
            return 20;
        case 3:
            return 30;
    }
    return 0;
}

int main() {
    return test_switch_return(2);
}
""", "Switch with return in cases (no break needed)")

write_test("cf059-switch-empty-case.cpp", """
int test_switch_empty(int x) {
    int result = 0;
    switch (x) {
        case 1:
        case 2:
        case 3:
            result = 123;
            break;
        case 4:
        case 5:
            result = 45;
            break;
        default:
            result = 0;
    }
    return result;
}

int main() {
    return test_switch_empty(2);
}
""", "Switch with multiple empty cases (grouped)")

write_test("cf060-switch-large-values.cpp", """
int test_switch_large(int x) {
    switch (x) {
        case 100:
            return 1;
        case 200:
            return 2;
        case 300:
            return 3;
        case 1000:
            return 4;
        default:
            return 0;
    }
}

int main() {
    return test_switch_large(300);
}
""", "Switch with large sparse case values")

# === GOTO STATEMENTS (10 tests) ===

write_test("cf061-simple-goto.cpp", """
int test_simple_goto() {
    int result = 0;
    goto skip;
    result = 10;
skip:
    result = 20;
    return result;
}

int main() {
    return test_simple_goto();
}
""", "Simple goto statement")

write_test("cf062-goto-forward.cpp", """
int test_goto_forward() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i == 5) goto end;
        sum += i;
    }
end:
    return sum;
}

int main() {
    return test_goto_forward();
}
""", "Forward goto to exit loop early")

write_test("cf063-goto-backward.cpp", """
int test_goto_backward() {
    int sum = 0;
    int i = 0;
start:
    sum += i;
    i++;
    if (i < 10) goto start;
    return sum;
}

int main() {
    return test_goto_backward();
}
""", "Backward goto (loop simulation)")

write_test("cf064-goto-cleanup-pattern.cpp", """
int test_goto_cleanup(int x) {
    int* ptr = nullptr;
    int result = -1;

    if (x < 0) goto cleanup;

    ptr = new int(42);
    if (x == 0) goto cleanup;

    result = *ptr;

cleanup:
    delete ptr;
    return result;
}

int main() {
    return test_goto_cleanup(1);
}
""", "Goto for cleanup pattern")

write_test("cf065-goto-nested-loops.cpp", """
int test_goto_nested() {
    int result = -1;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i == 5 && j == 3) {
                result = i * 10 + j;
                goto found;
            }
        }
    }
found:
    return result;
}

int main() {
    return test_goto_nested();
}
""", "Goto to break out of nested loops")

write_test("cf066-goto-multiple-labels.cpp", """
int test_multiple_gotos(int x) {
    if (x == 1) goto label1;
    if (x == 2) goto label2;
    if (x == 3) goto label3;
    return 0;

label1:
    return 10;
label2:
    return 20;
label3:
    return 30;
}

int main() {
    return test_multiple_gotos(2);
}
""", "Multiple goto targets")

write_test("cf067-goto-skip-initialization.cpp", """
int test_goto_skip_init(bool flag) {
    if (flag) goto skip;
    int x = 10;
    return x;
skip:
    return 20;
}

int main() {
    return test_goto_skip_init(true);
}
""", "Goto skipping variable initialization")

write_test("cf068-goto-error-handling.cpp", """
int test_goto_error(int x) {
    if (x < 0) goto error1;
    if (x == 0) goto error2;
    if (x > 100) goto error3;

    return x * 2;

error1:
    return -1;
error2:
    return -2;
error3:
    return -3;
}

int main() {
    return test_goto_error(50);
}
""", "Goto for error handling")

write_test("cf069-goto-state-machine.cpp", """
int test_goto_state_machine(int input) {
    int state = 0;
    int result = 0;

state0:
    if (input == 0) goto state1;
    goto end;

state1:
    result = 10;
    if (input == 0) goto state2;
    goto end;

state2:
    result = 20;

end:
    return result;
}

int main() {
    return test_goto_state_machine(0);
}
""", "Goto-based state machine")

write_test("cf070-goto-loop-continue.cpp", """
int test_goto_loop_continue() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) goto next;
        sum += i;
next:
        ;
    }
    return sum;
}

int main() {
    return test_goto_loop_continue();
}
""", "Goto simulating continue")

# === BREAK/CONTINUE (5 tests) ===

write_test("cf071-break-nested-loops.cpp", """
int test_break_nested() {
    int result = -1;
    bool found = false;
    for (int i = 0; i < 10 && !found; i++) {
        for (int j = 0; j < 10; j++) {
            if (i * j == 20) {
                result = i * 10 + j;
                found = true;
                break;
            }
        }
    }
    return result;
}

int main() {
    return test_break_nested();
}
""", "Break in nested loops with flag")

write_test("cf072-continue-pattern.cpp", """
int test_continue_pattern() {
    int sum = 0;
    for (int i = 0; i < 20; i++) {
        if (i < 5) continue;
        if (i > 15) continue;
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_continue_pattern();
}
""", "Multiple continue conditions")

write_test("cf073-break-switch-in-loop.cpp", """
int test_break_switch_loop() {
    int result = 0;
    for (int i = 0; i < 10; i++) {
        switch (i) {
            case 5:
                break;  // breaks switch, not loop
            default:
                result++;
        }
    }
    return result;
}

int main() {
    return test_break_switch_loop();
}
""", "Break in switch inside loop")

write_test("cf074-continue-while.cpp", """
int test_continue_while() {
    int sum = 0;
    int i = 0;
    while (i < 20) {
        i++;
        if (i % 3 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_continue_while();
}
""", "Continue in while loop")

write_test("cf075-break-do-while.cpp", """
int test_break_do_while() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
        if (sum > 20) break;
    } while (i < 100);
    return sum;
}

int main() {
    return test_break_do_while();
}
""", "Break in do-while loop")

# === RETURN STATEMENTS (5 tests) ===

write_test("cf076-multiple-returns.cpp", """
int test_multiple_returns(int x) {
    if (x < 0) return -1;
    if (x == 0) return 0;
    if (x < 10) return 1;
    if (x < 100) return 2;
    return 3;
}

int main() {
    return test_multiple_returns(50);
}
""", "Function with multiple return statements")

write_test("cf077-return-in-loop.cpp", """
int test_return_in_loop(int target) {
    for (int i = 0; i < 100; i++) {
        if (i == target) return i;
    }
    return -1;
}

int main() {
    return test_return_in_loop(42);
}
""", "Return inside loop")

write_test("cf078-return-complex-expr.cpp", """
int compute(int x, int y) {
    return (x + y) * (x - y) + x * y;
}

int main() {
    return compute(5, 3);
}
""", "Return with complex expression")

write_test("cf079-void-return.cpp", """
void test_void_return(int x, int& result) {
    if (x < 0) {
        result = -1;
        return;
    }
    if (x == 0) {
        result = 0;
        return;
    }
    result = x * 2;
}

int main() {
    int r = 0;
    test_void_return(10, r);
    return r;
}
""", "Void function with early returns")

write_test("cf080-return-reference.cpp", """
int global = 42;

int& test_return_reference() {
    return global;
}

int main() {
    int& ref = test_return_reference();
    ref = 100;
    return global;
}
""", "Return by reference")

# === SHORT-CIRCUIT EVALUATION (10 tests) ===

write_test("cf081-short-circuit-and.cpp", """
bool expensive_check(int& counter) {
    counter++;
    return true;
}

int test_short_circuit_and() {
    int counter = 0;
    bool result = (false && expensive_check(counter));
    return counter;  // Should be 0 (expensive_check not called)
}

int main() {
    return test_short_circuit_and();
}
""", "Short-circuit AND evaluation")

write_test("cf082-short-circuit-or.cpp", """
bool expensive_check(int& counter) {
    counter++;
    return false;
}

int test_short_circuit_or() {
    int counter = 0;
    bool result = (true || expensive_check(counter));
    return counter;  // Should be 0 (expensive_check not called)
}

int main() {
    return test_short_circuit_or();
}
""", "Short-circuit OR evaluation")

write_test("cf083-short-circuit-null-check.cpp", """
int test_null_check(int* ptr) {
    if (ptr && *ptr > 0) {
        return *ptr;
    }
    return 0;
}

int main() {
    int val = 42;
    return test_null_check(&val);
}
""", "Short-circuit null pointer check")

write_test("cf084-short-circuit-bounds-check.cpp", """
int test_bounds_check(int* arr, int size, int index) {
    if (index >= 0 && index < size && arr[index] > 0) {
        return arr[index];
    }
    return -1;
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    return test_bounds_check(arr, 5, 2);
}
""", "Short-circuit bounds checking")

write_test("cf085-short-circuit-complex.cpp", """
bool check1(int& c) { c += 1; return true; }
bool check2(int& c) { c += 10; return false; }
bool check3(int& c) { c += 100; return true; }

int test_complex_short_circuit() {
    int counter = 0;
    bool result = check1(counter) && check2(counter) && check3(counter);
    return counter;  // Should be 11 (check3 not called)
}

int main() {
    return test_complex_short_circuit();
}
""", "Complex short-circuit evaluation")

write_test("cf086-short-circuit-side-effects.cpp", """
int test_side_effects(int x) {
    int result = 0;
    (x > 0) && (result = 10, true);
    return result;
}

int main() {
    return test_side_effects(5);
}
""", "Short-circuit with side effects")

write_test("cf087-short-circuit-nested.cpp", """
int test_nested_short_circuit(int x, int y, int z) {
    if ((x > 0 && y > 0) || (z > 0)) {
        return 1;
    }
    return 0;
}

int main() {
    return test_nested_short_circuit(0, 0, 5);
}
""", "Nested short-circuit conditions")

write_test("cf088-short-circuit-array-access.cpp", """
int test_array_short_circuit(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size && arr && arr[i] != 0; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int arr[5] = {1, 2, 0, 4, 5};
    return test_array_short_circuit(arr, 5);
}
""", "Short-circuit in loop condition with array access")

write_test("cf089-short-circuit-function-calls.cpp", """
int func1() { return 0; }
int func2() { return 1; }
int func3() { return 2; }

int test_function_short_circuit() {
    return func1() || func2() || func3();
}

int main() {
    return test_function_short_circuit();
}
""", "Short-circuit with function calls")

write_test("cf090-short-circuit-ternary.cpp", """
int test_short_circuit_ternary(int x) {
    int counter = 0;
    int result = (x > 0) ? (counter++, 10) : (counter += 5, 20);
    return counter;
}

int main() {
    return test_short_circuit_ternary(5);
}
""", "Short-circuit in ternary operator")

# === COMPLEX CONTROL FLOW (10 tests) ===

write_test("cf091-mixed-loops.cpp", """
int test_mixed_loops() {
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        int j = 0;
        while (j < i) {
            sum += i * j;
            j++;
        }
    }
    return sum;
}

int main() {
    return test_mixed_loops();
}
""", "Mixed for and while loops")

write_test("cf092-loop-switch-combo.cpp", """
int test_loop_switch() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        switch (i % 3) {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 2;
                break;
            case 2:
                sum += 3;
                break;
        }
    }
    return sum;
}

int main() {
    return test_loop_switch();
}
""", "Loop with switch inside")

write_test("cf093-recursive-fibonacci.cpp", """
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    return fibonacci(10);
}
""", "Recursive function (Fibonacci)")

write_test("cf094-tail-recursion.cpp", """
int factorial_helper(int n, int acc) {
    if (n <= 1) return acc;
    return factorial_helper(n - 1, n * acc);
}

int factorial(int n) {
    return factorial_helper(n, 1);
}

int main() {
    return factorial(5);
}
""", "Tail recursion (factorial)")

write_test("cf095-mutual-recursion.cpp", """
int is_even(int n);
int is_odd(int n);

int is_even(int n) {
    if (n == 0) return 1;
    return is_odd(n - 1);
}

int is_odd(int n) {
    if (n == 0) return 0;
    return is_even(n - 1);
}

int main() {
    return is_even(10);
}
""", "Mutual recursion")

write_test("cf096-deeply-nested.cpp", """
int test_deeply_nested(int x) {
    if (x > 0) {
        if (x > 10) {
            if (x > 20) {
                if (x > 30) {
                    return 4;
                }
                return 3;
            }
            return 2;
        }
        return 1;
    }
    return 0;
}

int main() {
    return test_deeply_nested(25);
}
""", "Deeply nested if statements")

write_test("cf097-loop-with-multiple-exits.cpp", """
int test_multiple_exits(int target) {
    for (int i = 0; i < 100; i++) {
        if (i == target) return i;
        if (i * i > target) return -1;
        if (i % 10 == 0 && i > 50) break;
    }
    return 0;
}

int main() {
    return test_multiple_exits(42);
}
""", "Loop with multiple exit points")

write_test("cf098-early-exit-pattern.cpp", """
int test_early_exits(int x, int y, int z) {
    if (x < 0) return -1;
    if (y < 0) return -2;
    if (z < 0) return -3;
    if (x + y + z == 0) return 0;
    return x * y * z;
}

int main() {
    return test_early_exits(2, 3, 4);
}
""", "Early exit pattern (guard clauses)")

write_test("cf099-state-machine.cpp", """
enum State { INIT, PROCESSING, DONE, ERROR };

int test_state_machine(int input) {
    State state = INIT;
    int result = 0;

    while (state != DONE && state != ERROR) {
        switch (state) {
            case INIT:
                if (input > 0) {
                    state = PROCESSING;
                } else {
                    state = ERROR;
                }
                break;
            case PROCESSING:
                result = input * 2;
                state = DONE;
                break;
            default:
                state = ERROR;
        }
    }

    return (state == DONE) ? result : -1;
}

int main() {
    return test_state_machine(21);
}
""", "State machine pattern")

write_test("cf100-complex-branching.cpp", """
int test_complex_branching(int a, int b, int c) {
    int result = 0;

    if (a > b) {
        if (b > c) {
            result = a + b + c;
        } else if (a > c) {
            result = a + c;
        } else {
            result = c;
        }
    } else {
        if (a > c) {
            result = b + a;
        } else if (b > c) {
            result = b + c;
        } else {
            result = c;
        }
    }

    return result;
}

int main() {
    return test_complex_branching(5, 10, 3);
}
""", "Complex branching logic")

print(f"\nGeneration complete! Created {test_num-1} control flow tests.")
