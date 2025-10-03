// cls054-inherit.cpp
// Inheritance 54
// Test #454


class Base{ public: virtual int get(){return 54;} };
class Derived: public Base{ public: int get(){return 54+1;} };
int main(){ Derived d; return d.get(); }
