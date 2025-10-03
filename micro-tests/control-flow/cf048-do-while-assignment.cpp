// cf048-do-while-assignment.cpp
// Do-while with assignment
// Test #048


int get_value(int& counter) {
    return counter++;
}

int test_do_while_assignment() {
    int counter = 0;
    int sum = 0;
    int val;
    do {
        val = get_value(counter);
        sum += val;
    } while (val < 10);
    return sum;
}

int main() {
    return test_do_while_assignment();
}
