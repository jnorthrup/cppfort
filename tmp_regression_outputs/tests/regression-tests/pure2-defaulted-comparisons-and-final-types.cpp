#include <iostream>
#include "cpp2_inline.h"
void widget: final type =
{
    v: int;

    operator=(out this, int value) { v = value; }

    (this, that) -> bool operator==;

    (this, that) -> std::strong_ordering operator<=>; }

int main() { widget a = 1;
    widget b = 2;
    if a < b {
        std::cout << "less";
    }
    else {
        std::cout << "more";
    } }
