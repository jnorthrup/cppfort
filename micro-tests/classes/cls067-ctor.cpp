// cls067-ctor.cpp
// Constructor/destructor 67
// Test #467


class Test{ int x; public: Test():x(67){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
