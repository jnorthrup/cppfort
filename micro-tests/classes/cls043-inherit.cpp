// cls043-inherit.cpp
// Inheritance 43
// Test #443


class Base{ public: virtual int get(){return 43;} };
class Derived: public Base{ public: int get(){return 43+1;} };
int main(){ Derived d; return d.get(); }
