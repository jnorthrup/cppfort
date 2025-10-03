// cf059-switch-empty-case.cpp
// Switch with multiple empty cases (grouped)
// Test #059


int test_switch_empty(int x) {
    int result = 0;
    switch (x) {
        case 1:
        case 2:
        case 3:
            result = 123;
            break;
        case 4:
        case 5:
            result = 45;
            break;
        default:
            result = 0;
    }
    return result;
}

int main() {
    return test_switch_empty(2);
}
