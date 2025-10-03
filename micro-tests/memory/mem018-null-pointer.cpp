// mem018-null-pointer.cpp
// Null pointer check
// Test #098


int test_null_pointer() {
    int* ptr = nullptr;
    return (ptr == nullptr) ? 1 : 0;
}

int main() {
    return test_null_pointer();
}
