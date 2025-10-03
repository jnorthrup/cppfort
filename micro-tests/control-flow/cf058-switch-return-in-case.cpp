// cf058-switch-return-in-case.cpp
// Switch with return in cases (no break needed)
// Test #058


int test_switch_return(int x) {
    switch (x) {
        case 1:
            return 10;
        case 2:
            return 20;
        case 3:
            return 30;
    }
    return 0;
}

int main() {
    return test_switch_return(2);
}
