// cls032-inherit.cpp
// Inheritance 32
// Test #432


class Base{ public: virtual int get(){return 32;} };
class Derived: public Base{ public: int get(){return 32+1;} };
int main(){ Derived d; return d.get(); }
