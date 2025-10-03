// cf039-while-empty-body.cpp
// While loop with empty body
// Test #039


int test_while_empty() {
    int i = 0;
    while (i++ < 10);
    return i;
}

int main() {
    return test_while_empty();
}
