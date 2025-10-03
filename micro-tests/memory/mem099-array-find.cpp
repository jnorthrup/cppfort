// mem099-array-find.cpp
// Array linear search
// Test #179


int test_array_find(int target) {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

int main() {
    return test_array_find(3);
}
