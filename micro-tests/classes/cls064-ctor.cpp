// cls064-ctor.cpp
// Constructor/destructor 64
// Test #464


class Test{ int x; public: Test():x(64){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
