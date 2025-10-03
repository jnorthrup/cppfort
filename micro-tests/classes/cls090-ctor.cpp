// cls090-ctor.cpp
// Constructor/destructor 90
// Test #490


class Test{ int x; public: Test():x(90){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
