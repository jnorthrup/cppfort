// mem056-calloc.cpp
// Calloc (zero-initialized allocation)
// Test #136


#include <cstdlib>

int test_calloc() {
    int* arr = (int*)calloc(5, sizeof(int));
    arr[2] = 42;
    int result = arr[2];
    free(arr);
    return result;
}

int main() {
    return test_calloc();
}
