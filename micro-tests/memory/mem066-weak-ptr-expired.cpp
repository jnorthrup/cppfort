// mem066-weak-ptr-expired.cpp
// Weak pointer expired check
// Test #146


#include <memory>

int test_weak_ptr_expired() {
    std::weak_ptr<int> wptr;
    {
        std::shared_ptr<int> sptr = std::make_shared<int>(42);
        wptr = sptr;
    }
    return wptr.expired() ? 1 : 0;
}

int main() {
    return test_weak_ptr_expired();
}
