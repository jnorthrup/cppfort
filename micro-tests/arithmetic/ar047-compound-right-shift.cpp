// ar047-compound-right-shift.cpp
// Compound right shift (>>=)
// Test #047


int test_compound_rshift(int x, int n) {
    x >>= n;
    return x;
}

int main() {
    return test_compound_rshift(20, 2);
}
