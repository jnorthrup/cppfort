// cls053-inherit.cpp
// Inheritance 53
// Test #453


class Base{ public: virtual int get(){return 53;} };
class Derived: public Base{ public: int get(){return 53+1;} };
int main(){ Derived d; return d.get(); }
