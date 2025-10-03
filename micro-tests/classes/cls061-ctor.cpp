// cls061-ctor.cpp
// Constructor/destructor 61
// Test #461


class Test{ int x; public: Test():x(61){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
