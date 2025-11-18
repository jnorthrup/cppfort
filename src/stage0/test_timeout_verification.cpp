// Minimal test to verify a CTest TIMEOUT is enforced
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "timeout test: sleeping for 6 seconds\n";
    std::this_thread::sleep_for(std::chrono::seconds(6));
    std::cout << "timeout test: woke up (should be killed by CTest TIMEOUT)\n";
    return 0;
}
