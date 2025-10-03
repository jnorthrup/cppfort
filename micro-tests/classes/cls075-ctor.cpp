// cls075-ctor.cpp
// Constructor/destructor 75
// Test #475


class Test{ int x; public: Test():x(75){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
