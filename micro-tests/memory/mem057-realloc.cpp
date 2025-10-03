// mem057-realloc.cpp
// Realloc
// Test #137


#include <cstdlib>

int test_realloc() {
    int* arr = (int*)malloc(5 * sizeof(int));
    arr[2] = 42;
    arr = (int*)realloc(arr, 10 * sizeof(int));
    int result = arr[2];
    free(arr);
    return result;
}

int main() {
    return test_realloc();
}
