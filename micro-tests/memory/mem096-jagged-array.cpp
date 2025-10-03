// mem096-jagged-array.cpp
// Jagged array (array of arrays)
// Test #176


int test_jagged_array() {
    int* arr[3];
    int row0[2] = {1, 2};
    int row1[3] = {3, 4, 5};
    int row2[1] = {6};
    arr[0] = row0;
    arr[1] = row1;
    arr[2] = row2;
    return arr[1][1];
}

int main() {
    return test_jagged_array();
}
