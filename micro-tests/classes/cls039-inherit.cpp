// cls039-inherit.cpp
// Inheritance 39
// Test #439


class Base{ public: virtual int get(){return 39;} };
class Derived: public Base{ public: int get(){return 39+1;} };
int main(){ Derived d; return d.get(); }
