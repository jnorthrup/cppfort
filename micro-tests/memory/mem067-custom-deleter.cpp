// mem067-custom-deleter.cpp
// Smart pointer with custom deleter
// Test #147


#include <memory>

void custom_deleter(int* ptr) {
    delete ptr;
}

int test_custom_deleter() {
    std::shared_ptr<int> ptr(new int(42), custom_deleter);
    return *ptr;
}

int main() {
    return test_custom_deleter();
}
