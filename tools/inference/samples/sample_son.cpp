// Sample inspired by Sea-of-Nodes IR concepts
// This represents typical control flow patterns that map to SoN/MLIR regions

#include <iostream>

// Simple function with return
int constant_func() {
    return 42;
}

// Function with conditional
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Function with loop
int sum_to_n(int n) {
    int total = 0;
    for (int i = 1; i <= n; ++i) {
        total += i;
    }
    return total;
}

// Function with while loop
int factorial(int n) {
    int result = 1;
    while (n > 1) {
        result *= n;
        n--;
    }
    return result;
}

// Function with multiple returns
int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

// Nested control flow
int nested_example(int x, int y) {
    int result = 0;
    if (x > 0) {
        for (int i = 0; i < y; ++i) {
            if (i % 2 == 0) {
                result += i;
            }
        }
    }
    return result;
}
