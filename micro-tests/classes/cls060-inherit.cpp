// cls060-inherit.cpp
// Inheritance 60
// Test #460


class Base{ public: virtual int get(){return 60;} };
class Derived: public Base{ public: int get(){return 60+1;} };
int main(){ Derived d; return d.get(); }
