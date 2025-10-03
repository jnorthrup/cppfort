// cls057-inherit.cpp
// Inheritance 57
// Test #457


class Base{ public: virtual int get(){return 57;} };
class Derived: public Base{ public: int get(){return 57+1;} };
int main(){ Derived d; return d.get(); }
