// mem062-shared-ptr.cpp
// Shared pointer
// Test #142


#include <memory>

int test_shared_ptr() {
    std::shared_ptr<int> ptr = std::make_shared<int>(42);
    return *ptr;
}

int main() {
    return test_shared_ptr();
}
