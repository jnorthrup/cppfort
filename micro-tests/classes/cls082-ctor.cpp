// cls082-ctor.cpp
// Constructor/destructor 82
// Test #482


class Test{ int x; public: Test():x(82){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
