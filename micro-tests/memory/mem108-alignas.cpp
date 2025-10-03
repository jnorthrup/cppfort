// mem108-alignas.cpp
// Alignas specifier
// Test #188


struct alignas(16) Aligned {
    int x;
};

int test_alignas() {
    return alignof(Aligned);
}

int main() {
    return test_alignas();
}
