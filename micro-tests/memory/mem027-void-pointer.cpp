// mem027-void-pointer.cpp
// Void pointer casting
// Test #107


int test_void_pointer() {
    int x = 42;
    void* vptr = &x;
    int* ptr = (int*)vptr;
    return *ptr;
}

int main() {
    return test_void_pointer();
}
