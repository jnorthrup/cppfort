#include <iostream>
#include <vector>

int main() { std::cout <<   "while_continue_inner:\n  "; while_continue_inner();
    std::cout << "\nwhile_continue_outer:\n  "; while_continue_outer();
    std::cout << "\nwhile_break_inner:\n  ";    while_break_inner();
    std::cout << "\nwhile_break_outer:\n  ";    while_break_outer();

    std::cout <<  "\n\ndo_continue_inner:\n  "; do_continue_inner();
    std::cout <<    "\ndo_continue_outer:\n  "; do_continue_outer();
    std::cout <<    "\ndo_break_inner:\n  ";    do_break_inner();
    std::cout <<    "\ndo_break_outer:\n  ";    do_break_outer();

    std::cout << "\n\nfor_continue_inner:\n  "; for_continue_inner();
    std::cout <<   "\nfor_continue_outer:\n  "; for_continue_outer();
    std::cout <<   "\nfor_break_inner:\n  ";    for_break_inner();
    std::cout <<   "\nfor_break_outer:\n  ";    for_break_outer(); }

void while_continue_inner() { auto i = 0;
    auto while i<3 next i++ {
        j = 0;
        inner: while j<3 next j++ {
            std::cout << i << j << " ";
            if j == 1 {
                continue inner;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } }

void while_continue_outer() { auto i = 0;
    while i<3 next i++ {
        j : outer = 0;
        while j<3 next j++ {
            std::cout << i << j << " ";
            if j == 1 {
                continue outer;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } }

void while_break_inner() { auto i = 0;
    auto while i<3 next i++ {
        j = 0;
        inner: while j<3 next j++ {
            std::cout << i << j << " ";
            if j == 1 {
                break inner;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } }

void while_break_outer() { auto i = 0;
    while i<3 next i++ {
        j : outer = 0;
        while j<3 next j++ {
            std::cout << i << j << " ";
            if j == 1 {
                break outer;
            }
            std::cout << "inner ";
        }
        std::cout << "outer ";
    } }

void do_continue_inner() { auto i = 0;
    auto do {
        j = 0;
        inner: do {
            std::cout << i << j << " ";
            if j == 1 {
                continue inner;
            }
            std::cout << "inner ";
        }
        next j++ while j<3;

        std::cout << "outer ";
    }
    next i++ while i<3; }

void do_continue_outer() { auto i = 0;
    do {
        j : outer = 0;
        do {
            std::cout << i << j << " ";
            if j == 1 {
                continue outer;
            }
            std::cout << "inner ";
        }
        next j++ while j<3;

        std::cout << "outer ";
    }
    next i++ while i<3; }

void do_break_inner() { auto i = 0;
    auto do {
        j = 0;
        inner: do {
            std::cout << i << j << " ";
            if j == 1 {
                break inner;
            }
            std::cout << "inner ";
        }
        next j++ while j<3;

        std::cout << "outer ";
    }
    next i++ while i<3; }

void do_break_outer() { auto i = 0;
    do {
        j : outer = 0;
        do {
            std::cout << i << j << " ";
            if j == 1 {
                break outer;
            }
            std::cout << "inner ";
        }
        next j++ while j<3;

        std::cout << "outer ";
    }
    next i++ while i<3; }

void for_continue_inner() { std::vector vi = ( 0, 1, 2 );
    for vi do (i) {
        vj: std::vector = ( 0, 1, 2 );
        inner: for vj do (j) {
            std::cout << i << j << " ";
            if j == 1 {
                continue inner;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } }

void for_continue_outer() { std::vector vi = ( 0, 1, 2 );
    for vi do (i) {
        vj: std::vector outer = ( 0, 1, 2 );
        for vj do (j) {
            std::cout << i << j << " ";
            if j == 1 {
                continue outer;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } }

void for_break_inner() { std::vector vi = ( 0, 1, 2 );
    for vi do (i) {
        vj: std::vector = ( 0, 1, 2 );
        inner: for vj do (j) {
            std::cout << i << j << " ";
            if j == 1 {
                break inner;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } }

void for_break_outer() { std::vector vi = ( 0, 1, 2 );
    for vi do (i) {
        vj: std::vector outer = ( 0, 1, 2 );
        for vj do (j) {
            std::cout << i << j << " ";
            if j == 1 {
                break outer;
            }
            std::cout << "inner ";
        }

        std::cout << "outer ";
    } }
