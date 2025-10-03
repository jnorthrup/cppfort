// cf071-break-nested-loops.cpp
// Break in nested loops with flag
// Test #071


int test_break_nested() {
    int result = -1;
    bool found = false;
    for (int i = 0; i < 10 && !found; i++) {
        for (int j = 0; j < 10; j++) {
            if (i * j == 20) {
                result = i * 10 + j;
                found = true;
                break;
            }
        }
    }
    return result;
}

int main() {
    return test_break_nested();
}
