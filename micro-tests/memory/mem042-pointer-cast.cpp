// mem042-pointer-cast.cpp
// Pointer type casting
// Test #122


int test_pointer_cast() {
    int x = 0x12345678;
    char* ptr = (char*)&x;
    return ptr[0];
}

int main() {
    return test_pointer_cast();
}
