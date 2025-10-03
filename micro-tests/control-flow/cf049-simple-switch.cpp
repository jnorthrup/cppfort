// cf049-simple-switch.cpp
// Simple switch statement
// Test #049


int test_simple_switch(int x) {
    switch (x) {
        case 1:
            return 10;
        case 2:
            return 20;
        case 3:
            return 30;
        default:
            return 0;
    }
}

int main() {
    return test_simple_switch(2);
}
