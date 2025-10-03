// cls058-inherit.cpp
// Inheritance 58
// Test #458


class Base{ public: virtual int get(){return 58;} };
class Derived: public Base{ public: int get(){return 58+1;} };
int main(){ Derived d; return d.get(); }
