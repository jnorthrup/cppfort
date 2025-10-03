// cf070-goto-loop-continue.cpp
// Goto simulating continue
// Test #070


int test_goto_loop_continue() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) goto next;
        sum += i;
next:
        ;
    }
    return sum;
}

int main() {
    return test_goto_loop_continue();
}
