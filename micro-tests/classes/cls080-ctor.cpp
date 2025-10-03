// cls080-ctor.cpp
// Constructor/destructor 80
// Test #480


class Test{ int x; public: Test():x(80){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
