// mem115-anonymous-union.cpp
// Anonymous union in struct
// Test #195


struct Data {
    union {
        int i;
        float f;
    };
};

int test_anonymous_union() {
    Data d;
    d.i = 42;
    return d.i;
}

int main() {
    return test_anonymous_union();
}
