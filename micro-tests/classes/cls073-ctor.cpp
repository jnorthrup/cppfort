// cls073-ctor.cpp
// Constructor/destructor 73
// Test #473


class Test{ int x; public: Test():x(73){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
