// mem118-volatile-pointer.cpp
// Volatile pointer
// Test #198


int test_volatile_pointer() {
    volatile int x = 42;
    volatile int* ptr = &x;
    return *ptr;
}

int main() {
    return test_volatile_pointer();
}
