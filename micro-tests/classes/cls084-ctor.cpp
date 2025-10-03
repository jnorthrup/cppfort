// cls084-ctor.cpp
// Constructor/destructor 84
// Test #484


class Test{ int x; public: Test():x(84){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
