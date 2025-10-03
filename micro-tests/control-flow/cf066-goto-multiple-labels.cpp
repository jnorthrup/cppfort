// cf066-goto-multiple-labels.cpp
// Multiple goto targets
// Test #066


int test_multiple_gotos(int x) {
    if (x == 1) goto label1;
    if (x == 2) goto label2;
    if (x == 3) goto label3;
    return 0;

label1:
    return 10;
label2:
    return 20;
label3:
    return 30;
}

int main() {
    return test_multiple_gotos(2);
}
