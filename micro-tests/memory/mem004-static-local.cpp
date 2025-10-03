// mem004-static-local.cpp
// Static local variable
// Test #084


int test_static_local() {
    static int counter = 0;
    return ++counter;
}

int main() {
    test_static_local();
    return test_static_local();
}
