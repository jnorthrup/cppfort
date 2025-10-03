// cls083-ctor.cpp
// Constructor/destructor 83
// Test #483


class Test{ int x; public: Test():x(83){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
