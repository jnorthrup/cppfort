// cf042-do-while-execute-once.cpp
// Do-while that executes exactly once
// Test #042


int test_do_while_once() {
    int sum = 0;
    do {
        sum = 42;
    } while (false);
    return sum;
}

int main() {
    return test_do_while_once();
}
