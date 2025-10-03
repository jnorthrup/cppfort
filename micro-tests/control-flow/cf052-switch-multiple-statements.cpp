// cf052-switch-multiple-statements.cpp
// Switch cases with multiple statements
// Test #052


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
