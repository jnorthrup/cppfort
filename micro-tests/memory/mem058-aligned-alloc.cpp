// mem058-aligned-alloc.cpp
// Aligned allocation
// Test #138


#include <cstdlib>

int test_aligned_alloc() {
    int* ptr = (int*)aligned_alloc(64, sizeof(int));
    if (!ptr) return -1;
    *ptr = 42;
    int result = *ptr;
    free(ptr);
    return result;
}

int main() {
    return test_aligned_alloc();
}
