// cf065-goto-nested-loops.cpp
// Goto to break out of nested loops
// Test #065


int test_goto_nested() {
    int result = -1;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (i == 5 && j == 3) {
                result = i * 10 + j;
                goto found;
            }
        }
    }
found:
    return result;
}

int main() {
    return test_goto_nested();
}
