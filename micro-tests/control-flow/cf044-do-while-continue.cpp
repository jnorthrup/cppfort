// cf044-do-while-continue.cpp
// Do-while with continue
// Test #044


int test_do_while_continue() {
    int sum = 0;
    int i = 0;
    do {
        i++;
        if (i % 2 == 0) continue;
        sum += i;
    } while (i < 10);
    return sum;
}

int main() {
    return test_do_while_continue();
}
