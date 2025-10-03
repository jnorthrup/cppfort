// cls087-ctor.cpp
// Constructor/destructor 87
// Test #487


class Test{ int x; public: Test():x(87){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
