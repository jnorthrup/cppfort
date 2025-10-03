// ar015-compound-div.cpp
// Compound division (/=)
// Test #015


int test_compound_div(int x, int y) {
    if (y == 0) return 0;
    x /= y;
    return x;
}

int main() {
    return test_compound_div(20, 4);
}
