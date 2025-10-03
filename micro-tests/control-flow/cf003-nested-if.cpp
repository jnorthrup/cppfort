// cf003-nested-if.cpp
// Nested if statements
// Test #003


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
