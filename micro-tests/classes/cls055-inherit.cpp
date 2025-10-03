// cls055-inherit.cpp
// Inheritance 55
// Test #455


class Base{ public: virtual int get(){return 55;} };
class Derived: public Base{ public: int get(){return 55+1;} };
int main(){ Derived d; return d.get(); }
