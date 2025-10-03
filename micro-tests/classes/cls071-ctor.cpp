// cls071-ctor.cpp
// Constructor/destructor 71
// Test #471


class Test{ int x; public: Test():x(71){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
