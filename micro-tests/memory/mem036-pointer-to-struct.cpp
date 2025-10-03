// mem036-pointer-to-struct.cpp
// Pointer to struct with arrow operator
// Test #116


struct Point { int x; int y; };

int test_pointer_to_struct() {
    Point p = {3, 4};
    Point* ptr = &p;
    return ptr->x + ptr->y;
}

int main() {
    return test_pointer_to_struct();
}
