// mem039-dangling-pointer.cpp
// Dangling pointer (undefined behavior)
// Test #119


int* dangling_pointer() {
    int x = 42;
    return &x;  // Dangling pointer!
}

int main() {
    // This is undefined behavior, but we'll return a safe value
    return 0;
}
