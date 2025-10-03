// mem028-const-pointer.cpp
// Pointer to const
// Test #108


int test_const_pointer() {
    int x = 42;
    const int* ptr = &x;
    return *ptr;
}

int main() {
    return test_const_pointer();
}
