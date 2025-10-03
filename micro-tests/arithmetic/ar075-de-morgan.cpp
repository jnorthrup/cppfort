// ar075-de-morgan.cpp
// De Morgan's laws
// Test #075


int test_de_morgan(bool a, bool b) {
    return !(a && b) == (!a || !b);
}

int main() {
    return test_de_morgan(true, false);
}
