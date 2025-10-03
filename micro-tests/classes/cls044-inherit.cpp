// cls044-inherit.cpp
// Inheritance 44
// Test #444


class Base{ public: virtual int get(){return 44;} };
class Derived: public Base{ public: int get(){return 44+1;} };
int main(){ Derived d; return d.get(); }
