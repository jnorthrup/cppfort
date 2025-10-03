// cf073-break-switch-in-loop.cpp
// Break in switch inside loop
// Test #073


int test_break_switch_loop() {
    int result = 0;
    for (int i = 0; i < 10; i++) {
        switch (i) {
            case 5:
                break;  // breaks switch, not loop
            default:
                result++;
        }
    }
    return result;
}

int main() {
    return test_break_switch_loop();
}
