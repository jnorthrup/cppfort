// cf079-void-return.cpp
// Void function with early returns
// Test #079


void test_void_return(int x, int& result) {
    if (x < 0) {
        result = -1;
        return;
    }
    if (x == 0) {
        result = 0;
        return;
    }
    result = x * 2;
}

int main() {
    int r = 0;
    test_void_return(10, r);
    return r;
}
