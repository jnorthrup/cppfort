// cls048-inherit.cpp
// Inheritance 48
// Test #448


class Base{ public: virtual int get(){return 48;} };
class Derived: public Base{ public: int get(){return 48+1;} };
int main(){ Derived d; return d.get(); }
