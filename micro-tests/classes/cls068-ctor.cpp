// cls068-ctor.cpp
// Constructor/destructor 68
// Test #468


class Test{ int x; public: Test():x(68){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
