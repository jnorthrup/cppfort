// cf064-goto-cleanup-pattern.cpp
// Goto for cleanup pattern
// Test #064


int test_goto_cleanup(int x) {
    int* ptr = nullptr;
    int result = -1;

    if (x < 0) goto cleanup;

    ptr = new int(42);
    if (x == 0) goto cleanup;

    result = *ptr;

cleanup:
    delete ptr;
    return result;
}

int main() {
    return test_goto_cleanup(1);
}
