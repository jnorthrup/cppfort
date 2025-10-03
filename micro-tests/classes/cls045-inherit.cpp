// cls045-inherit.cpp
// Inheritance 45
// Test #445


class Base{ public: virtual int get(){return 45;} };
class Derived: public Base{ public: int get(){return 45+1;} };
int main(){ Derived d; return d.get(); }
