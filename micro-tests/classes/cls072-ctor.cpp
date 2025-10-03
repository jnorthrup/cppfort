// cls072-ctor.cpp
// Constructor/destructor 72
// Test #472


class Test{ int x; public: Test():x(72){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
