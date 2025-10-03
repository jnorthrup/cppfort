// ar005-mod-int.cpp
// Integer modulo
// Test #005


int test_mod(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}

int main() {
    return test_mod(17, 5);
}
