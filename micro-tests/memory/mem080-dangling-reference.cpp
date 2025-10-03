// mem080-dangling-reference.cpp
// Dangling reference (undefined behavior)
// Test #160


int& dangling_reference() {
    int x = 42;
    return x;  // Dangling reference!
}

int main() {
    // Undefined behavior, return safe value
    return 0;
}
