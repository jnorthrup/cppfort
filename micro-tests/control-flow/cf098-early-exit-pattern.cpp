// cf098-early-exit-pattern.cpp
// Early exit pattern (guard clauses)
// Test #098


int test_early_exits(int x, int y, int z) {
    if (x < 0) return -1;
    if (y < 0) return -2;
    if (z < 0) return -3;
    if (x + y + z == 0) return 0;
    return x * y * z;
}

int main() {
    return test_early_exits(2, 3, 4);
}
