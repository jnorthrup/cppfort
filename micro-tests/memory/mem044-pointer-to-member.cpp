// mem044-pointer-to-member.cpp
// Pointer to member variable
// Test #124


struct Data {
    int x;
    int y;
};

int test_pointer_to_member() {
    Data d = {3, 4};
    int Data::*ptr = &Data::x;
    return d.*ptr;
}

int main() {
    return test_pointer_to_member();
}
