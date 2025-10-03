// cls074-ctor.cpp
// Constructor/destructor 74
// Test #474


class Test{ int x; public: Test():x(74){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
