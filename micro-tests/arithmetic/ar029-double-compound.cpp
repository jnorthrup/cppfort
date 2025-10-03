// ar029-double-compound.cpp
// Double compound operations
// Test #029


double test_double_compound(double x, double y) {
    x += y;
    x *= 2.0;
    return x;
}

int main() {
    return (int)test_double_compound(3.0, 2.0);
}
