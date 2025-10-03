// mem116-volatile-read.cpp
// Volatile variable read
// Test #196


volatile int global_volatile = 42;

int test_volatile_read() {
    return global_volatile;
}

int main() {
    return test_volatile_read();
}
