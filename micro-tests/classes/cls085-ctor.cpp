// cls085-ctor.cpp
// Constructor/destructor 85
// Test #485


class Test{ int x; public: Test():x(85){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
