// cf023-for-empty-body.cpp
// For loop with empty body
// Test #023


int test_for_empty() {
    int i;
    for (i = 0; i < 10; i++);
    return i;
}

int main() {
    return test_for_empty();
}
