#include <iostream>

int main(args) { auto local_int = 42;

    auto //  'in' (read-only) statement scope variable
    (i = local_int) for args do (_) {
        std::cout << i << "\n";     // prints 42
    }

    auto //  'inout' (read-write) statement scope variable
    (inout i = local_int) {
        i++;
    }
    std::cout << local_int << "\n";     std::chrono::duration<i32> // prints 43

    _ = 0;
    (copy _: std::chrono::duration<i32> = 0) { }
    std::in_place_t _ = ();
    (copy _: std::in_place_t = ()) { } }
