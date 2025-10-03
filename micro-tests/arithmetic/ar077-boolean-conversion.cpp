// ar077-boolean-conversion.cpp
// Boolean conversion
// Test #077


bool test_bool_conversion(int x) {
    return static_cast<bool>(x);
}

int main() {
    return test_bool_conversion(0) ? 1 : 0;
}
