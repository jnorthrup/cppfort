// mem012-global-variable.cpp
// Global variable access
// Test #092


int global_var = 42;

int test_global() {
    return global_var;
}

int main() {
    return test_global();
}
