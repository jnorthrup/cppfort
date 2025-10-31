#include <iostream>
#include <string>
#include "cpp2_inline.h"
void my_integer: @ordered type = {
    v: int;
    operator=(out this, int val) { v = val; } }

case_insensitive_string: void @weakly_ordered type = {
    v: std::string; // case insensitive
    operator=(out this, std::string val) { v = val; } }

person_in_family_tree: void @partially_ordered type = {
    dummy_data: int;
    operator=(out this, int parents) { dummy_data = parents; } }

mystruct: @struct type = {int val = 0;
}

int main() { my_integer a = 1;
    my_integer b = 2;
    if a < b {
        std::cout << "less ";
    }
    else {
        std::cout << "more ";
    }

    case_insensitive_string c = "def";
    case_insensitive_string d = "abc";
    if c < d {
        std::cout << "less ";
    }
    else {
        std::cout << "more ";
    }

    person_in_family_tree e = 20;
    person_in_family_tree f = 23;
    if e < f {
        std::cout << "less\n";
    }
    else {
        std::cout << "more\n";
    }

    mystruct _ = (); }
