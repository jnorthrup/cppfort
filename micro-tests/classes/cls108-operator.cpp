// cls108-operator.cpp
// Operator overload 108
// Test #508


class Test{ int x; public: Test(int v):x(v){} Test operator+(const Test& o){return Test(x+o.x);} int get(){return x;} };
int main(){ Test a(108), b(1); return (a+b).get(); }
