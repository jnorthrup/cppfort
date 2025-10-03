// cf092-loop-switch-combo.cpp
// Loop with switch inside
// Test #092


int test_loop_switch() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        switch (i % 3) {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 2;
                break;
            case 2:
                sum += 3;
                break;
        }
    }
    return sum;
}

int main() {
    return test_loop_switch();
}
