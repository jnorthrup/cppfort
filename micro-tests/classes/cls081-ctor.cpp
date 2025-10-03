// cls081-ctor.cpp
// Constructor/destructor 81
// Test #481


class Test{ int x; public: Test():x(81){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
