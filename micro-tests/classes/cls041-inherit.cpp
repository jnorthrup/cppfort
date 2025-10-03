// cls041-inherit.cpp
// Inheritance 41
// Test #441


class Base{ public: virtual int get(){return 41;} };
class Derived: public Base{ public: int get(){return 41+1;} };
int main(){ Derived d; return d.get(); }
