// ar027-float-comparison.cpp
// Float comparison
// Test #027


int test_float_compare(float a, float b) {
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

int main() {
    return test_float_compare(3.5f, 2.5f);
}
