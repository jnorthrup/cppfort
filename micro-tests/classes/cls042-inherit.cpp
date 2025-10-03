// cls042-inherit.cpp
// Inheritance 42
// Test #442


class Base{ public: virtual int get(){return 42;} };
class Derived: public Base{ public: int get(){return 42+1;} };
int main(){ Derived d; return d.get(); }
