// ar069-abs-function.cpp
// Absolute value function
// Test #069


int test_abs(int x) {
    return (x < 0) ? -x : x;
}

int main() {
    return test_abs(-5);
}
