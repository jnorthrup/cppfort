// cls040-inherit.cpp
// Inheritance 40
// Test #440


class Base{ public: virtual int get(){return 40;} };
class Derived: public Base{ public: int get(){return 40+1;} };
int main(){ Derived d; return d.get(); }
