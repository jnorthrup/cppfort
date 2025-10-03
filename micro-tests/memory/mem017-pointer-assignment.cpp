// mem017-pointer-assignment.cpp
// Assignment through pointer
// Test #097


int test_pointer_assignment() {
    int x = 10;
    int* ptr = &x;
    *ptr = 20;
    return x;
}

int main() {
    return test_pointer_assignment();
}
