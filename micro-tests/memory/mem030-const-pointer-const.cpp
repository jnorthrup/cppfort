// mem030-const-pointer-const.cpp
// Const pointer to const
// Test #110


int test_const_pointer_const() {
    int x = 42;
    const int* const ptr = &x;
    return *ptr;
}

int main() {
    return test_const_pointer_const();
}
