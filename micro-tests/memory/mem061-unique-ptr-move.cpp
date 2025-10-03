// mem061-unique-ptr-move.cpp
// Unique pointer move semantics
// Test #141


#include <memory>

int test_unique_ptr_move() {
    std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    return *ptr2;
}

int main() {
    return test_unique_ptr_move();
}
