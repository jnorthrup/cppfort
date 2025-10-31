#include <iostream>
#include <string>
void general_regex_test: @regex type = {
  regex_01 := R"(AA)";
  regex_02 := R"((?=aa))";
  run(this) { std::cout << "Running tests_01_char_matcher:"<< std::endl;

    " << regex_01.match("AAaa").matched << std::endl std::cout << "Not full match fails;
    " << regex_01.match("AA").matched << std::endl std::cout << "Full match is ok;
    " << regex_01.search("aAAaa").group_start(0) << std::endl std::cout << "Search finds at position 1;
    " << regex_01.search("aaaAAaa").group_start(0) << std::endl std::cout << "Search finds at position 3;

    auto count = 0;
    auto func = :(r) -> bool == {
      count&$* += 1;
      std::cout << "Find all finds at position: " << r.group_start(0) << std::endl;
      return true;
    };

    std::string str = "aAAaAAaAAa";
    regex_01..find_all(func, str);
    " << count << std::endl std::cout << "Find all found 3 matched;

    count = 0;
    str = "bbaabb";
    regex_02..find_all(func, str);
    " << count << std::endl std::cout << "Find all found 1 match;
  } }

int main() { general_regex_test().run(); }
