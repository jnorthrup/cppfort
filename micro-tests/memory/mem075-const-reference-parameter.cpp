// mem075-const-reference-parameter.cpp
// Const reference parameter
// Test #155


int get_value(const int& x) {
    return x;
}

int main() {
    return get_value(42);
}
