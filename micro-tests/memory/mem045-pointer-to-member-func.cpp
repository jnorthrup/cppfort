// mem045-pointer-to-member-func.cpp
// Pointer to member function
// Test #125


struct Calculator {
    int add(int a, int b) { return a + b; }
};

int test_pointer_to_member_func() {
    Calculator calc;
    int (Calculator::*fptr)(int, int) = &Calculator::add;
    return (calc.*fptr)(3, 4);
}

int main() {
    return test_pointer_to_member_func();
}
