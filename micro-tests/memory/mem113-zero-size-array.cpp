// mem113-zero-size-array.cpp
// Flexible array member (zero-size)
// Test #193


struct FlexibleArray {
    int count;
    int data[];
};

int test_flexible_array() {
    return sizeof(FlexibleArray);
}

int main() {
    return test_flexible_array();
}
