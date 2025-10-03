// cls093-operator.cpp
// Operator overload 93
// Test #493


class Test{ int x; public: Test(int v):x(v){} Test operator+(const Test& o){return Test(x+o.x);} int get(){return x;} };
int main(){ Test a(93), b(1); return (a+b).get(); }
