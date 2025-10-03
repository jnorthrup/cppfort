// cls065-ctor.cpp
// Constructor/destructor 65
// Test #465


class Test{ int x; public: Test():x(65){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
