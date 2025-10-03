// mem083-reference-swap.cpp
// Swap using references
// Test #163


void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(x, y);
    return x;
}
