// Simple sample for inference prototype
int add(int a, int b) {
    int s = a + b;
    if (s > 10) {
        s -= 10;
    }
    for (int i = 0; i < 3; ++i) {
        s += i;
    }
    return s;
}
