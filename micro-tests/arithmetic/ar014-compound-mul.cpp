// ar014-compound-mul.cpp
// Compound multiplication (*=)
// Test #014


int test_compound_mul(int x, int y) {
    x *= y;
    return x;
}

int main() {
    return test_compound_mul(5, 4);
}
