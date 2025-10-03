// ar035-float-precision.cpp
// Float precision issues
// Test #035


double test_precision() {
    double x = 0.1;
    double y = 0.2;
    double z = 0.3;
    return (x + y == z) ? 1 : 0;
}

int main() {
    return test_precision();
}
