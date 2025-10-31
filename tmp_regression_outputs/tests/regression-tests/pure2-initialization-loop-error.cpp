
void f() { int i;
    while true {
        i = 42;     // ERROR: can't initialize i in a loop
    }
    i = 42; }
