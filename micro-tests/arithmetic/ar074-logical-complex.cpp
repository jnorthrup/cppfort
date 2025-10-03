// ar074-logical-complex.cpp
// Complex logical expression
// Test #074


int test_logical_complex(bool a, bool b, bool c) {
    return (a && b) || (!c);
}

int main() {
    return test_logical_complex(false, true, false);
}
