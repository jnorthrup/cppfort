// cls070-ctor.cpp
// Constructor/destructor 70
// Test #470


class Test{ int x; public: Test():x(70){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
