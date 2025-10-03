// cls069-ctor.cpp
// Constructor/destructor 69
// Test #469


class Test{ int x; public: Test():x(69){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
