// mem054-delete-null.cpp
// Delete null pointer (safe)
// Test #134


int test_delete_null() {
    int* ptr = nullptr;
    delete ptr;  // Safe: deleting null is no-op
    return 0;
}

int main() {
    return test_delete_null();
}
