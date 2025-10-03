// cf031-simple-while.cpp
// Simple while loop
// Test #031


int test_simple_while() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_simple_while();
}
