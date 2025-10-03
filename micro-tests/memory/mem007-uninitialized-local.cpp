// mem007-uninitialized-local.cpp
// Uninitialized then assigned local
// Test #087


int test_uninitialized() {
    int x;
    x = 42;
    return x;
}

int main() {
    return test_uninitialized();
}
