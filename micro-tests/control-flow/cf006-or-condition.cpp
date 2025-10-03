// cf006-or-condition.cpp
// If with logical OR condition
// Test #006


int test_or_condition(int x, int y) {
    if (x > 10 || y > 10) {
        return 1;
    }
    return 0;
}

int main() {
    return test_or_condition(5, 15);
}
