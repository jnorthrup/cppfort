// cf050-switch-fallthrough.cpp
// Switch with fall-through cases
// Test #050


int test_switch_fallthrough(int x) {
    int result = 0;
    switch (x) {
        case 1:
            result += 1;
        case 2:
            result += 2;
        case 3:
            result += 3;
            break;
        default:
            result = -1;
    }
    return result;
}

int main() {
    return test_switch_fallthrough(1);
}
