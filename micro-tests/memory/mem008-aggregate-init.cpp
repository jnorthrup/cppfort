// mem008-aggregate-init.cpp
// Aggregate initialization
// Test #088


struct Point { int x; int y; };

int test_aggregate_init() {
    Point p = {3, 4};
    return p.x + p.y;
}

int main() {
    return test_aggregate_init();
}
