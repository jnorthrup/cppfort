#include "src/selfhost/cpp2.h2"
#include <iostream>
int main(){
  std::string_view src = "a: int =\n42;";
  auto marks = cpp2::scan(src);
  std::cout << "marks=" << marks.size() << "\n";
  for (auto const& m: marks) std::cout << m.pos << ':' << static_cast<int>(m.k) << "\n";
  auto regs = cpp2::fold(src, marks);
  std::cout << "regions=" << regs.size() << "\n";
  for (auto const& r: regs) std::cout << static_cast<int>(r.k) << ' ' << r.lo << ' ' << r.hi << " => [" << src.substr(r.lo, r.hi-r.lo) << "]\n";
}
