// cf043-do-while-break.cpp
// Do-while with break
// Test #043


int test_do_while_break() {
    int sum = 0;
    int i = 0;
    do {
        if (i >= 10) break;
        sum += i;
        i++;
    } while (true);
    return sum;
}

int main() {
    return test_do_while_break();
}
