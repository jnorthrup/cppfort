fun0:  () -> forward _ = { i: int = 42; return i;  } // error: a 'forward' return type cannot return a local variable


//  These are no longer errors now that '-> forward _' can deduce
//  by-value return, but let's keep these cases as comments for now

// fun1:  () -> forward _ = { return global   / 1; } // error: a 'forward' return type cannot return a temporary variable
// fun4:  () -> forward _ = { return global  << 1; } // error: a 'forward' return type cannot return a temporary variable
// fun5:  () -> forward _ = { return global <=> 1; } // error: a 'forward' return type cannot return a temporary variable
// fun6:  () -> forward _ = { return global   < 1; } // error: a 'forward' return type cannot return a temporary variable
// fun7:  () -> forward _ = { return global  <= 1; } // error: a 'forward' return type cannot return a temporary variable
// fun9:  () -> forward _ = { return global  >> 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun10() { return global  >= 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun11() { return global   > 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun14() { return global   + 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun15() { return global   - 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun20() { return global  || 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun22() { return global   | 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun24() { return global  && 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun26() { return global   * 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun28() { return global   % 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun30() { return global   & 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun32() { return global   ^ 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun35() { return global  == 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun37() { return global  != 1; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun38() { return !ptr_g; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun42() { return global&; } // error: a 'forward' return type cannot return a temporary variable
auto&& // fun43() { return ptr_g~; } // error: a 'forward' return type cannot return a temporary variable

// fun2:  () -> forward _ = { return global  /= 1; }
// fun3:  () -> forward _ = { return global <<= 1; }
// fun8:  () -> forward _ = { return global >>= 1; }
auto&& // fun12() { return global++; }
auto&& // fun13() { return global  += 1; }
auto&& // fun16() { return global  -= 1; }
auto&& // fun17() { return ptr_g*.x; }
auto&& // fun18() { return global--; }
auto&& // fun21() { return global  |= 1; }
auto&& // fun25() { return global  *= 1; }
auto&& // fun27() { return global  %= 1; }
auto&& // fun29() { return global  &= 1; }
auto&& // fun31() { return global  ^= 1; }
auto&& // fun36() { return global   = 1; }
auto&& // fun39() { return g.x; }
auto&& // fun41() { return ptr_g*; }

auto&& // fun19() { return global ||= 1; } // not supported
auto&& // fun23() { return global &&= 1; } // not supported
auto&& // fun33() { return global  ~= 1; } // not supported
auto&& // fun34() { return global   ~ 1; } // not supported
auto&& // fun40() { g return ptr_g ? ptr_g*; } // not supported


global: i32 = 42;using t = {
    public x : int = 42;
};t g = ();
*t ptr_g = g&;
