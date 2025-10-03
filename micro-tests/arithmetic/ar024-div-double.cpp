// ar024-div-double.cpp
// Double division
// Test #024


double test_div_double(double a, double b) {
    if (b == 0.0) return 0.0;
    return a / b;
}

int main() {
    return (int)test_div_double(20.0, 4.0);
}
