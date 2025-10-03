// cls046-inherit.cpp
// Inheritance 46
// Test #446


class Base{ public: virtual int get(){return 46;} };
class Derived: public Base{ public: int get(){return 46+1;} };
int main(){ Derived d; return d.get(); }
