// mem052-memory-leak.cpp
// Memory leak (no delete)
// Test #132


int test_memory_leak() {
    int* ptr = new int(42);
    // Intentional leak (no delete)
    return *ptr;
}

int main() {
    return test_memory_leak();
}
