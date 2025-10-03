// cls062-ctor.cpp
// Constructor/destructor 62
// Test #462


class Test{ int x; public: Test():x(62){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
