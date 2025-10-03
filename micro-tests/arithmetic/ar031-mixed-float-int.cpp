// ar031-mixed-float-int.cpp
// Mixed int and float arithmetic
// Test #031


float test_mixed_types(int a, float b) {
    return a + b;
}

int main() {
    return (int)test_mixed_types(5, 3.5f);
}
