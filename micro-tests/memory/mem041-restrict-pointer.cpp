// mem041-restrict-pointer.cpp
// Restrict pointer qualifier
// Test #121


int test_restrict(int* __restrict a, int* __restrict b) {
    *a = 10;
    *b = 20;
    return *a;
}

int main() {
    int x = 0, y = 0;
    return test_restrict(&x, &y);
}
