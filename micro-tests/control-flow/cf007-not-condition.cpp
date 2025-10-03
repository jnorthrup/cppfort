// cf007-not-condition.cpp
// If with logical NOT condition
// Test #007


int test_not_condition(bool flag) {
    if (!flag) {
        return 1;
    }
    return 0;
}

int main() {
    return test_not_condition(false);
}
