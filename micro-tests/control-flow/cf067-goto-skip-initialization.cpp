// cf067-goto-skip-initialization.cpp
// Goto skipping variable initialization
// Test #067


int test_goto_skip_init(bool flag) {
    if (flag) goto skip;
    int x = 10;
    return x;
skip:
    return 20;
}

int main() {
    return test_goto_skip_init(true);
}
