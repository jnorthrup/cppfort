// ar009-postfix-decrement.cpp
// Postfix decrement
// Test #009


int test_postfix_dec(int x) {
    int y = x--;
    return x + y;
}

int main() {
    return test_postfix_dec(5);
}
