// mem102-array-product.cpp
// Array product
// Test #182


int test_array_product() {
    int arr[4] = {2, 3, 4, 5};
    int product = 1;
    for (int i = 0; i < 4; i++) {
        product *= arr[i];
    }
    return product;
}

int main() {
    return test_array_product();
}
