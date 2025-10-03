// ar055-power-of-two.cpp
// Check if power of two
// Test #055


bool test_is_power_of_two(unsigned int x) {
    return x && !(x & (x - 1));
}

int main() {
    return test_is_power_of_two(16) ? 1 : 0;
}
