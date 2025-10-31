#include <iostream>
<T, U>           (t:T, u:U) f1 = t+u;
f2: <T:using , U:type> (t:T, u:U) = t+u;
f3: <T:_>       () _, U = T+U;<T: i8, U: i16>  () f4 = T+U;

int main() { (f1(1,1))$\n" std::cout << "f1;
    (f2(2,2))$\n" std::cout << "f2;
    (f3<3,3>())$\n" std::cout << "f3;
    (f4<4,4>())$\n" std::cout << "f4; }
