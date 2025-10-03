// cf038-while-assignment.cpp
// While loop with assignment in condition
// Test #038


int get_value(int& counter) {
    return counter++;
}

int test_while_assignment() {
    int counter = 0;
    int sum = 0;
    int val;
    while ((val = get_value(counter)) < 10) {
        sum += val;
    }
    return sum;
}

int main() {
    return test_while_assignment();
}
