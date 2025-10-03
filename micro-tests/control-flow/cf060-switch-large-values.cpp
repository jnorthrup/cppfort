// cf060-switch-large-values.cpp
// Switch with large sparse case values
// Test #060


int test_switch_large(int x) {
    switch (x) {
        case 100:
            return 1;
        case 200:
            return 2;
        case 300:
            return 3;
        case 1000:
            return 4;
        default:
            return 0;
    }
}

int main() {
    return test_switch_large(300);
}
