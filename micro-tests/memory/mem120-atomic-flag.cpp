// mem120-atomic-flag.cpp
// Atomic flag operations
// Test #200


#include <atomic>

std::atomic_flag flag = ATOMIC_FLAG_INIT;

int test_atomic_flag() {
    bool was_set = flag.test_and_set();
    flag.clear();
    return was_set ? 1 : 0;
}

int main() {
    return test_atomic_flag();
}
