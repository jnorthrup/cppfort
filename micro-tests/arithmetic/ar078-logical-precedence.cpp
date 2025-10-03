// ar078-logical-precedence.cpp
// Logical operator precedence
// Test #078


int test_logical_precedence(bool a, bool b, bool c) {
    return a || b && c;
}

int main() {
    return test_logical_precedence(false, true, true);
}
