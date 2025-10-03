// ar068-clamp-function.cpp
// Clamp function
// Test #068


int test_clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

int main() {
    return test_clamp(15, 0, 10);
}
