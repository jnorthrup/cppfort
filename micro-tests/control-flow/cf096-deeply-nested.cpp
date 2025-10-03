// cf096-deeply-nested.cpp
// Deeply nested if statements
// Test #096


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
