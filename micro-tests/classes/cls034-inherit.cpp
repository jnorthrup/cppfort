// cls034-inherit.cpp
// Inheritance 34
// Test #434


class Base{ public: virtual int get(){return 34;} };
class Derived: public Base{ public: int get(){return 34+1;} };
int main(){ Derived d; return d.get(); }
