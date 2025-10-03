// mem074-reference-parameter.cpp
// Reference parameter
// Test #154


void increment(int& x) {
    x++;
}

int main() {
    int value = 41;
    increment(value);
    return value;
}
