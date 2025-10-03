// mem014-static-global.cpp
// Static global variable
// Test #094


static int static_global = 10;

int test_static_global() {
    return static_global;
}

int main() {
    return test_static_global();
}
