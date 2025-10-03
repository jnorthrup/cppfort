// cls056-inherit.cpp
// Inheritance 56
// Test #456


class Base{ public: virtual int get(){return 56;} };
class Derived: public Base{ public: int get(){return 56+1;} };
int main(){ Derived d; return d.get(); }
