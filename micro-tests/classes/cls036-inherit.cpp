// cls036-inherit.cpp
// Inheritance 36
// Test #436


class Base{ public: virtual int get(){return 36;} };
class Derived: public Base{ public: int get(){return 36+1;} };
int main(){ Derived d; return d.get(); }
