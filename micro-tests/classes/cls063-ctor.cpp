// cls063-ctor.cpp
// Constructor/destructor 63
// Test #463


class Test{ int x; public: Test():x(63){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
