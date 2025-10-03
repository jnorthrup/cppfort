// cf097-loop-with-multiple-exits.cpp
// Loop with multiple exit points
// Test #097


int test_multiple_exits(int target) {
    for (int i = 0; i < 100; i++) {
        if (i == target) return i;
        if (i * i > target) return -1;
        if (i % 10 == 0 && i > 50) break;
    }
    return 0;
}

int main() {
    return test_multiple_exits(42);
}
