#include <iostream>
void fun(v) { if v is (pred_i) {
        std::cout << "(v)$ is integer bigger than 3" << std::endl;
    }

    if v is (pred_d) {
        std::cout << "(v)$ is double bigger than 3" << std::endl;
    }

    if v is (pred_) {
        std::cout << "(v)$ is bigger than 3" << std::endl;
    } }

int main() { fun(3.14);
    fun(42);
    fun('a'); }

bool pred_i(int x) { return x > 3; }

bool pred_d(double x) { return x > 3; }

bool pred_(x) { return x > 3; }
