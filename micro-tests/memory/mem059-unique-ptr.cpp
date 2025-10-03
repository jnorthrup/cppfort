// mem059-unique-ptr.cpp
// Unique pointer (RAII)
// Test #139


#include <memory>

int test_unique_ptr() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    return *ptr;
}

int main() {
    return test_unique_ptr();
}
