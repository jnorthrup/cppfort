// mem051-nothrow-new.cpp
// Nothrow new
// Test #131


#include <new>

int test_nothrow_new() {
    int* ptr = new (std::nothrow) int(42);
    if (!ptr) return -1;
    int result = *ptr;
    delete ptr;
    return result;
}

int main() {
    return test_nothrow_new();
}
