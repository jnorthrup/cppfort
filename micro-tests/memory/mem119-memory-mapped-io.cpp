// mem119-memory-mapped-io.cpp
// Memory-mapped I/O pattern
// Test #199


#define HARDWARE_REG ((volatile unsigned int*)0x40000000)

int test_mmio() {
    // Simulated memory-mapped I/O
    // *HARDWARE_REG = 0x12345678;
    // return *HARDWARE_REG;
    return 42;  // Placeholder
}

int main() {
    return test_mmio();
}
