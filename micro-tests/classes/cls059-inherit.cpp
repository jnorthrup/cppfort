// cls059-inherit.cpp
// Inheritance 59
// Test #459


class Base{ public: virtual int get(){return 59;} };
class Derived: public Base{ public: int get(){return 59+1;} };
int main(){ Derived d; return d.get(); }
