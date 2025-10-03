// cf037-while-pointer.cpp
// While loop with pointer iteration
// Test #037


int test_while_pointer() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    int sum = 0;
    while (ptr < arr + 5) {
        sum += *ptr;
        ptr++;
    }
    return sum;
}

int main() {
    return test_while_pointer();
}
