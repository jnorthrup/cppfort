// mem050-placement-new.cpp
// Placement new
// Test #130


#include <new>

int test_placement_new() {
    char buffer[sizeof(int)];
    int* ptr = new (buffer) int(42);
    int result = *ptr;
    ptr->~int();
    return result;
}

int main() {
    return test_placement_new();
}
