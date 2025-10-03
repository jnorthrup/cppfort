// cls050-inherit.cpp
// Inheritance 50
// Test #450


class Base{ public: virtual int get(){return 50;} };
class Derived: public Base{ public: int get(){return 50+1;} };
int main(){ Derived d; return d.get(); }
