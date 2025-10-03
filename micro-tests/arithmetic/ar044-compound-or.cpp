// ar044-compound-or.cpp
// Compound bitwise OR (|=)
// Test #044


int test_compound_or(int x, int y) {
    x |= y;
    return x;
}

int main() {
    return test_compound_or(0b1100, 0b0011);
}
