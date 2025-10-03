// cls086-ctor.cpp
// Constructor/destructor 86
// Test #486


class Test{ int x; public: Test():x(86){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
