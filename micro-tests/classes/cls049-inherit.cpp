// cls049-inherit.cpp
// Inheritance 49
// Test #449


class Base{ public: virtual int get(){return 49;} };
class Derived: public Base{ public: int get(){return 49+1;} };
int main(){ Derived d; return d.get(); }
