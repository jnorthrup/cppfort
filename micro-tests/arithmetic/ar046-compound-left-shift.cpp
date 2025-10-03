// ar046-compound-left-shift.cpp
// Compound left shift (<<=)
// Test #046


int test_compound_lshift(int x, int n) {
    x <<= n;
    return x;
}

int main() {
    return test_compound_lshift(5, 2);
}
