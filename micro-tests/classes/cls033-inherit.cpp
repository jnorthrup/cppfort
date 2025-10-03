// cls033-inherit.cpp
// Inheritance 33
// Test #433


class Base{ public: virtual int get(){return 33;} };
class Derived: public Base{ public: int get(){return 33+1;} };
int main(){ Derived d; return d.get(); }
