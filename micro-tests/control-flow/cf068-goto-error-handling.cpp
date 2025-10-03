// cf068-goto-error-handling.cpp
// Goto for error handling
// Test #068


int test_goto_error(int x) {
    if (x < 0) goto error1;
    if (x == 0) goto error2;
    if (x > 100) goto error3;

    return x * 2;

error1:
    return -1;
error2:
    return -2;
error3:
    return -3;
}

int main() {
    return test_goto_error(50);
}
