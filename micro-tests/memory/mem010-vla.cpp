// mem010-vla.cpp
// Variable-length array (VLA)
// Test #090


int test_vla(int n) {
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }
    return arr[n-1];
}

int main() {
    return test_vla(5);
}
