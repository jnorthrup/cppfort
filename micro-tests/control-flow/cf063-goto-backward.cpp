// cf063-goto-backward.cpp
// Backward goto (loop simulation)
// Test #063


int test_goto_backward() {
    int sum = 0;
    int i = 0;
start:
    sum += i;
    i++;
    if (i < 10) goto start;
    return sum;
}

int main() {
    return test_goto_backward();
}
