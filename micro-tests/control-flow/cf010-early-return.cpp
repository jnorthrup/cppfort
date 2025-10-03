// cf010-early-return.cpp
// Multiple early return guards
// Test #010


int test_early_return(int x) {
    if (x < 0) {
        return 0;
    }
    if (x > 100) {
        return 100;
    }
    return x;
}

int main() {
    return test_early_return(50);
}
