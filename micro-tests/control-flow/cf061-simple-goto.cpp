// cf061-simple-goto.cpp
// Simple goto statement
// Test #061


int test_simple_goto() {
    int result = 0;
    goto skip;
    result = 10;
skip:
    result = 20;
    return result;
}

int main() {
    return test_simple_goto();
}
