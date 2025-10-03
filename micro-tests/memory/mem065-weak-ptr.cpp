// mem065-weak-ptr.cpp
// Weak pointer
// Test #145


#include <memory>

int test_weak_ptr() {
    std::shared_ptr<int> sptr = std::make_shared<int>(42);
    std::weak_ptr<int> wptr = sptr;
    std::shared_ptr<int> sptr2 = wptr.lock();
    return sptr2 ? *sptr2 : -1;
}

int main() {
    return test_weak_ptr();
}
