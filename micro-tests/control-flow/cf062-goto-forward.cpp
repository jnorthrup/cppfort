// cf062-goto-forward.cpp
// Forward goto to exit loop early
// Test #062


int test_goto_forward() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i == 5) goto end;
        sum += i;
    }
end:
    return sum;
}

int main() {
    return test_goto_forward();
}
