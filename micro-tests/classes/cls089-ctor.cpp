// cls089-ctor.cpp
// Constructor/destructor 89
// Test #489


class Test{ int x; public: Test():x(89){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
