// cls088-ctor.cpp
// Constructor/destructor 88
// Test #488


class Test{ int x; public: Test():x(88){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
