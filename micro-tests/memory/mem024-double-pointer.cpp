// mem024-double-pointer.cpp
// Double pointer (pointer to pointer)
// Test #104


int test_double_pointer() {
    int x = 42;
    int* ptr = &x;
    int** pptr = &ptr;
    return **pptr;
}

int main() {
    return test_double_pointer();
}
