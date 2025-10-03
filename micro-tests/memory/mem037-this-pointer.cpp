// mem037-this-pointer.cpp
// This pointer in member function
// Test #117


class Counter {
    int value;
public:
    Counter(int v) : value(v) {}
    int getValue() { return this->value; }
};

int main() {
    Counter c(42);
    return c.getValue();
}
