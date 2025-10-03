// ar032-long-double.cpp
// Long double arithmetic
// Test #032


long double test_long_double(long double x, long double y) {
    return x * y;
}

int main() {
    return (int)test_long_double(3.0L, 4.0L);
}
