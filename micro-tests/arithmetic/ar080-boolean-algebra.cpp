// ar080-boolean-algebra.cpp
// Boolean algebra distributive law
// Test #080


bool test_boolean_algebra(bool a, bool b, bool c) {
    return (a && (b || c)) == ((a && b) || (a && c));
}

int main() {
    return test_boolean_algebra(true, false, true);
}
