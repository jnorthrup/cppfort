// cf008-multi-statement-if.cpp
// If block with multiple statements
// Test #008


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
