// cf056-switch-range.cpp
// Switch with GCC range extension
// Test #056


int test_switch_range(int x) {
    switch (x) {
        case 0 ... 9:
            return 1;
        case 10 ... 19:
            return 2;
        default:
            return 0;
    }
}

int main() {
    return test_switch_range(15);
}
