// cls078-ctor.cpp
// Constructor/destructor 78
// Test #478


class Test{ int x; public: Test():x(78){} ~Test(){} int get(){return x;} };
int main(){ Test t; return t.get(); }
