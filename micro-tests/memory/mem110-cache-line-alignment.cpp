// mem110-cache-line-alignment.cpp
// Cache line alignment
// Test #190


struct alignas(64) CacheLine {
    int data[16];
};

int test_cache_line() {
    CacheLine cl;
    cl.data[0] = 42;
    return cl.data[0];
}

int main() {
    return test_cache_line();
}
