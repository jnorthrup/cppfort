// mem117-volatile-write.cpp
// Volatile variable write
// Test #197


volatile int global_volatile = 0;

void test_volatile_write(int value) {
    global_volatile = value;
}

int main() {
    test_volatile_write(42);
    return global_volatile;
}
