// cls037-inherit.cpp
// Inheritance 37
// Test #437


class Base{ public: virtual int get(){return 37;} };
class Derived: public Base{ public: int get(){return 37+1;} };
int main(){ Derived d; return d.get(); }
