// cf077-return-in-loop.cpp
// Return inside loop
// Test #077


int test_return_in_loop(int target) {
    for (int i = 0; i < 100; i++) {
        if (i == target) return i;
    }
    return -1;
}

int main() {
    return test_return_in_loop(42);
}
