// mem094-array-copy.cpp
// Array copy
// Test #174


int test_array_copy() {
    int src[3] = {1, 2, 3};
    int dest[3];
    for (int i = 0; i < 3; i++) {
        dest[i] = src[i];
    }
    return dest[1];
}

int main() {
    return test_array_copy();
}
