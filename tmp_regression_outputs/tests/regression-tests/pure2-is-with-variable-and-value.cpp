#include <iostream>
#include <string>
#include <vector>
void fun(name, v) { :cout << name << ": " <<
                inspect v -> std::string {
                is (42) std = "42";
                is (123) = "op_is";
                is (-123) = "generic op_is";
                is (4321) = "comparable";
                is ("text") = "text";
                is _ = "unknown";
              }
              << std::endl; }

int main() { fun("3.14", 3.14);
    fun("42", 42);
    fun("WithOp()", WithOp());
    fun("WithGenOp()", WithGenOp());
    fun("Cmp()", Cmp());
    fun("std::string(\"text\")", std::string("text"));
    fun("\"text\"", "text");
    fun("std::string_view(\"text\")", std::string_view("text"));
    fun(":std::vector = ('t','e','x','t','\\0')", :std::vector = ('t','e','x','t','\0')); }auto WithOp : type = {
    op_is(this, int x) { return x == 123;
}; }

WithGenOp : bool type = {
    op_is(this, x) { :convertible_to<decltype(x), int> {
            return x if constexpr std = = -123;
        }
        return false;
    } }

Cmp : bool type = {
    operator==(this, int x) { return x == 4321;
}; }
