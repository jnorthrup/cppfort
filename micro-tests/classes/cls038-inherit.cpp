// cls038-inherit.cpp
// Inheritance 38
// Test #438


class Base{ public: virtual int get(){return 38;} };
class Derived: public Base{ public: int get(){return 38+1;} };
int main(){ Derived d; return d.get(); }
