// mem046-new-delete.cpp
// Basic new and delete
// Test #126


int test_new_delete() {
    int* ptr = new int(42);
    int result = *ptr;
    delete ptr;
    return result;
}

int main() {
    return test_new_delete();
}
