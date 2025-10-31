#include <iostream>
void A: type = {
  public i: int = 0;

	const_foo(virtual this) { std::cout << "const foo \n"; }
	auto mut_foo = [](inout this) { std::cout << "foo \n"; }; }

B: type = {A this = ();double public d = 0.0;
}

void func_mut(A& a) { (a.i)$" << std::endl std::cout << "Call A mut; }
void func_mut(B& b) { (b.d)$" << std::endl std::cout << "Call B mut; }
void func_const(A a) { (a.i)$" << std::endl std::cout << "Call A const; }
void func_const(B b) { (b.d)$" << std::endl std::cout << "Call B const; }

void test_const_foo() { A s = ();
  *const A sC = s&;
  s.const_foo();
  sC*.const_foo();
	(s as A).const_foo();
  (sC* as A).const_foo();
  _ = s;
  _ = sC; }

void test_mut_foo() { A s = ();
  s.mut_foo();
	(s as A).mut_foo();
  _ = s; }

void test_up() { B b = ();
  *const B bC = b&;

  func_const(b);
  func_const(b as B);
  func_const(b as A);
  func_const(bC*);
  func_const(bC* as B);
  func_const(bC* as A);

  func_mut(b);
  func_mut(b as B);
  func_mut(b as A);

  _ = b;
  _ = bC; }

void test_down() { B b = ();
  *const B bC = b&;
  *A a = (b as A)&;
  *const  A aC = (b as A)&;

  func_const(a*);
  func_const(a* as B);
  func_const(a* as A);
  func_const(aC*);
  func_const(aC* as B);
  func_const(aC* as A);
  func_mut(a*);
  func_mut(a* as B);
  func_mut(a* as A);

  _ = b;
  _ = bC;
  _ = a;
  _ = aC; }

int main() { test_const_foo();
  test_mut_foo();
  test_up();
  test_down();

  return 0; }
