// cls027-basic.cpp
// Basic class 27
// Test #427


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(27); return t.get(); }
