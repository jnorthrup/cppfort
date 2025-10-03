// ar013-compound-sub.cpp
// Compound subtraction (-=)
// Test #013


int test_compound_sub(int x, int y) {
    x -= y;
    return x;
}

int main() {
    return test_compound_sub(10, 3);
}
