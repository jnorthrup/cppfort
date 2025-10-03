// cf013-ternary-assignment.cpp
// Ternary operator in assignment
// Test #013


int test_ternary_assignment(int x) {
    int result = (x > 0) ? x * 2 : x / 2;
    return result;
}

int main() {
    return test_ternary_assignment(10);
}
