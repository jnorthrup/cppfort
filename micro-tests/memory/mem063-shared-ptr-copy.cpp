// mem063-shared-ptr-copy.cpp
// Shared pointer copy (reference counting)
// Test #143


#include <memory>

int test_shared_ptr_copy() {
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::shared_ptr<int> ptr2 = ptr1;
    return *ptr2;
}

int main() {
    return test_shared_ptr_copy();
}
