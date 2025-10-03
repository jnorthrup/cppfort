// cls066-ctor.cpp
// Constructor/destructor 66
// Test #466


class Test{ int x; public: Test():x(66){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
