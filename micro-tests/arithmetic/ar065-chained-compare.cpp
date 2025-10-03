// ar065-chained-compare.cpp
// Chained comparisons
// Test #065


int test_chained_compare(int a, int b, int c) {
    return (a < b) && (b < c);
}

int main() {
    return test_chained_compare(1, 5, 10);
}
