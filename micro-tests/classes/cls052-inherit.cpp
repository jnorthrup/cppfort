// cls052-inherit.cpp
// Inheritance 52
// Test #452


class Base{ public: virtual int get(){return 52;} };
class Derived: public Base{ public: int get(){return 52+1;} };
int main(){ Derived d; return d.get(); }
