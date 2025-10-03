// cf094-tail-recursion.cpp
// Tail recursion (factorial)
// Test #094


int factorial_helper(int n, int acc) {
    if (n <= 1) return acc;
    return factorial_helper(n - 1, n * acc);
}

int factorial(int n) {
    return factorial_helper(n, 1);
}

int main() {
    return factorial(5);
}
