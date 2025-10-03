// mem048-new-struct.cpp
// New and delete struct
// Test #128


struct Point { int x; int y; };

int test_new_struct() {
    Point* ptr = new Point{3, 4};
    int result = ptr->x + ptr->y;
    delete ptr;
    return result;
}

int main() {
    return test_new_struct();
}
