// mem111-union-type-punning.cpp
// Union type punning
// Test #191


union Converter {
    int i;
    float f;
};

int test_union_punning() {
    Converter c;
    c.f = 3.14f;
    return c.i != 0;
}

int main() {
    return test_union_punning();
}
