// ar016-compound-mod.cpp
// Compound modulo (%=)
// Test #016


int test_compound_mod(int x, int y) {
    if (y == 0) return 0;
    x %= y;
    return x;
}

int main() {
    return test_compound_mod(17, 5);
}
