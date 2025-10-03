// cls076-ctor.cpp
// Constructor/destructor 76
// Test #476


class Test{ int x; public: Test():x(76){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
