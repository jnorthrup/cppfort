// cf040-while-flag-pattern.cpp
// While loop with boolean flag pattern
// Test #040


int test_while_flag() {
    bool found = false;
    int i = 0;
    int result = -1;
    while (!found && i < 10) {
        if (i == 5) {
            found = true;
            result = i;
        }
        i++;
    }
    return result;
}

int main() {
    return test_while_flag();
}
