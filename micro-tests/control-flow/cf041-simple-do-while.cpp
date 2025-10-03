// cf041-simple-do-while.cpp
// Simple do-while loop
// Test #041


int test_do_while() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
    } while (i < 10);
    return sum;
}

int main() {
    return test_do_while();
}
