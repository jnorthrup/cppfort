// mem064-shared-ptr-use-count.cpp
// Shared pointer use count
// Test #144


#include <memory>

int test_shared_ptr_use_count() {
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::shared_ptr<int> ptr2 = ptr1;
    return ptr1.use_count();
}

int main() {
    return test_shared_ptr_use_count();
}
