// cf100-complex-branching.cpp
// Complex branching logic
// Test #100


int test_complex_branching(int a, int b, int c) {
    int result = 0;

    if (a > b) {
        if (b > c) {
            result = a + b + c;
        } else if (a > c) {
            result = a + c;
        } else {
            result = c;
        }
    } else {
        if (a > c) {
            result = b + a;
        } else if (b > c) {
            result = b + c;
        } else {
            result = c;
        }
    }

    return result;
}

int main() {
    return test_complex_branching(5, 10, 3);
}
