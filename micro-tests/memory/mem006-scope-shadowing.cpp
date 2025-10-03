// mem006-scope-shadowing.cpp
// Variable shadowing in nested scope
// Test #086


int test_shadowing() {
    int x = 10;
    {
        int x = 20;
        return x;
    }
}

int main() {
    return test_shadowing();
}
