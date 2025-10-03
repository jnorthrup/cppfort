// cf004-if-else-if.cpp
// If-else-if chain
// Test #004


int test_if_else_if(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else if (x < 10) {
        return 1;
    } else {
        return 2;
    }
}

int main() {
    return test_if_else_if(5);
}
