// cf091-mixed-loops.cpp
// Mixed for and while loops
// Test #091


int test_mixed_loops() {
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        int j = 0;
        while (j < i) {
            sum += i * j;
            j++;
        }
    }
    return sum;
}

int main() {
    return test_mixed_loops();
}
