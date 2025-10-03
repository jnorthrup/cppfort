// mem053-double-delete.cpp
// Double delete (undefined behavior)
// Test #133


int test_double_delete() {
    int* ptr = new int(42);
    int result = *ptr;
    delete ptr;
    // delete ptr;  // Undefined behavior!
    return result;
}

int main() {
    return test_double_delete();
}
