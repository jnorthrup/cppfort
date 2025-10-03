// mem081-reference-to-pointer.cpp
// Reference to pointer
// Test #161


int test_reference_to_pointer() {
    int x = 42;
    int* ptr = &x;
    int*& ref = ptr;
    return *ref;
}

int main() {
    return test_reference_to_pointer();
}
