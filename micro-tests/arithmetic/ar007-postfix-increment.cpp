// ar007-postfix-increment.cpp
// Postfix increment
// Test #007


int test_postfix_inc(int x) {
    int y = x++;
    return x + y;
}

int main() {
    return test_postfix_inc(5);
}
