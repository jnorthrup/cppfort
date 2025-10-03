// cls031-inherit.cpp
// Inheritance 31
// Test #431


class Base{ public: virtual int get(){return 31;} };
class Derived: public Base{ public: int get(){return 31+1;} };
int main(){ Derived d; return d.get(); }
