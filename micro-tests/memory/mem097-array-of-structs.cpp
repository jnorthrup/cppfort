// mem097-array-of-structs.cpp
// Array of structs
// Test #177


struct Point { int x; int y; };

int test_array_of_structs() {
    Point arr[3] = {{1, 2}, {3, 4}, {5, 6}};
    return arr[1].x + arr[1].y;
}

int main() {
    return test_array_of_structs();
}
