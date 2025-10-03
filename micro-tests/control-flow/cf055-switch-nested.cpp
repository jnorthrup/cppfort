// cf055-switch-nested.cpp
// Nested switch statements
// Test #055


int test_nested_switch(int x, int y) {
    switch (x) {
        case 1:
            switch (y) {
                case 1: return 11;
                case 2: return 12;
                default: return 10;
            }
        case 2:
            return 20;
        default:
            return 0;
    }
}

int main() {
    return test_nested_switch(1, 2);
}
