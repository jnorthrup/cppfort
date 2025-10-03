// cls051-inherit.cpp
// Inheritance 51
// Test #451


class Base{ public: virtual int get(){return 51;} };
class Derived: public Base{ public: int get(){return 51+1;} };
int main(){ Derived d; return d.get(); }
