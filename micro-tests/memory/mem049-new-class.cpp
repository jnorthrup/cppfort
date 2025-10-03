// mem049-new-class.cpp
// New and delete class object
// Test #129


class Counter {
    int value;
public:
    Counter(int v) : value(v) {}
    int getValue() { return value; }
};

int test_new_class() {
    Counter* ptr = new Counter(42);
    int result = ptr->getValue();
    delete ptr;
    return result;
}

int main() {
    return test_new_class();
}
