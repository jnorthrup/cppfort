// mem033-function-pointer.cpp
// Function pointer
// Test #113


int add(int a, int b) { return a + b; }

int test_function_pointer() {
    int (*fptr)(int, int) = add;
    return fptr(3, 4);
}

int main() {
    return test_function_pointer();
}
