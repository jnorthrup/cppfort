#include <iostream>
#include "cpp2_inline.h"
void widget: @value type = {
    val: int = 0;
    operator=(out this, int i) { val = i; } }

w_widget: void @weakly_ordered_value type = {
    val: int = 0;
    operator=(out this, int i) { val = i; } }

p_widget: void @partially_ordered_value type = {
    val: int = 0;
    operator=(out this, int i) { val = i; } }

int main() { test<widget>();
    test<w_widget>();
    test<p_widget>(); }

test: <T> () = {T //  should be default constructible
    a = ();T //  widget should be comparable
    b = 2;
    if (a<b) {
        std::cout << "less ";
    }
    else {
        std::cout << "more ";
    }
}
