// cls035-inherit.cpp
// Inheritance 35
// Test #435


class Base{ public: virtual int get(){return 35;} };
class Derived: public Base{ public: int get(){return 35+1;} };
int main(){ Derived d; return d.get(); }
