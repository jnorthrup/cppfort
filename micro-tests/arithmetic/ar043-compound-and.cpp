// ar043-compound-and.cpp
// Compound bitwise AND (&=)
// Test #043


int test_compound_and(int x, int y) {
    x &= y;
    return x;
}

int main() {
    return test_compound_and(0b1111, 0b1010);
}
