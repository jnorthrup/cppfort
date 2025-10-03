// mem043-offsetof-pattern.cpp
// Manual offset calculation (offsetof pattern)
// Test #123


struct Data {
    int a;
    int b;
    int c;
};

int test_offsetof() {
    Data d = {1, 2, 3};
    char* ptr = (char*)&d;
    int* b_ptr = (int*)(ptr + sizeof(int));
    return *b_ptr;
}

int main() {
    return test_offsetof();
}
