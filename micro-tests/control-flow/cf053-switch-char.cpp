// cf053-switch-char.cpp
// Switch on char type
// Test #053


int test_switch_char(char c) {
    switch (c) {
        case 'a':
            return 1;
        case 'b':
            return 2;
        case 'c':
            return 3;
        default:
            return 0;
    }
}

int main() {
    return test_switch_char('b');
}
