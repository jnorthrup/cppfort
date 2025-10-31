#include <iostream>
using A = {};

VA: @polymorphic_base <I:int> type = {}using VC = {
    this: VA<0>;
    this: VA<1>;
};using VD = {
    this: VA<0>;
};

void fun(v, name) { std::cout << "(name)$ is";
    if v is VC    { std::cout <<  " VC"; }
    if v is VA<0> { std::cout <<  " VA<0>"; }
    if v is VA<1> { std::cout <<  " VA<1>"; }
    if v is VD    { std::cout <<  " VD"; }
    if v is *VC    { std::cout <<  " *VC"; }
    if v is *VA<0> { std::cout <<  " *VA<0>"; }
    if v is *VA<1> { std::cout <<  " *VA<1>"; }
    if v is *VD    { std::cout <<  " *VD"; }
    if v is void  { std::cout <<  " empty"; }
    if v is A  { std::cout <<  " A"; }
    std::cout << std::endl; }

int main() { VC vc = ();
    *VA<0> p0 = vc&;
    *VA<1> p1 = vc&;

    fun(vc, "vc");
    fun(p0*, "p0*");
    fun(p1*, "p1*");

    fun(vc&, "vc&");
    fun(p0, "p0");
    fun(p1, "p1");

    fun(nullptr, "nullptr");
    fun(A(), "A"); }