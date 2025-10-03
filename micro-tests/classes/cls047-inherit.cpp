// cls047-inherit.cpp
// Inheritance 47
// Test #447


class Base{ public: virtual int get(){return 47;} };
class Derived: public Base{ public: int get(){return 47+1;} };
int main(){ Derived d; return d.get(); }
