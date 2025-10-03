// mem055-malloc-free.cpp
// Malloc and free
// Test #135


#include <cstdlib>

int test_malloc_free() {
    int* ptr = (int*)malloc(sizeof(int));
    *ptr = 42;
    int result = *ptr;
    free(ptr);
    return result;
}

int main() {
    return test_malloc_free();
}
