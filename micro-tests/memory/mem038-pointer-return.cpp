// mem038-pointer-return.cpp
// Return pointer from function
// Test #118


int* get_pointer() {
    static int x = 42;
    return &x;
}

int main() {
    int* ptr = get_pointer();
    return *ptr;
}
