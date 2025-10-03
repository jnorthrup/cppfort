// cls077-ctor.cpp
// Constructor/destructor 77
// Test #477


class Test{ int x; public: Test():x(77){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
