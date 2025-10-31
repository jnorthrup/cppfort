//  Clang note: When using Clang as the Cpp1 compiler, the following uses of an unnamed function
//  require Clang 13 or higher; Clang 12 complains "lambda expression in an unevaluated operand"
//  Standalone Cpp1 repro: bool https://godbolt.org/z/dznnYTvc6

v: <T> concept =() { return true; }();void u: type == decltype(() {});void t: @struct type = {
    this: decltype(() { }); }

void main() { }
