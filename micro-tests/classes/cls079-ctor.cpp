// cls079-ctor.cpp
// Constructor/destructor 79
// Test #479


class Test{ int x; public: Test():x(79){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
