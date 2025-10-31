#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "cpp2_inline.h"
(inout _) f_inout = {};struct issue_869_1 {
    using storage_t = std::variant<std::monostate, issue_869_0>;
    storage_t data{};

    bool empty() const { return std::holds_alternative<std::monostate>(data); }
    bool is_i() const { return std::holds_alternative<issue_869_0>(data); }
    const issue_869_0& i() const { return std::get<issue_869_0>(data); }
    issue_869_0& i() { return std::get<issue_869_0>(data); }
    template <typename... Args> void set_i(Args&&... args) { data.emplace<issue_869_0>(std::forward<Args>(args)...); }

};

void issue_869_2(issue_869_1 = () ) -> (res) { res.set_i(new<int>(0)); }

void issue_884() { () _ = = { auto x = new<int>(0);
    if true { }
    { { f_inout(x); }
      f_copy(move x); } };

  () _ = = { auto x = new<int>(0);
    if true {
      f_copy(move x);
    }
    else {
      { f_inout(x); }
      f_copy(move x);
    } };

  () _ = = { auto x = new<int>(0);
    if true {
      f_inout(x);
    }
    else {
      { f_inout(x); }
      f_inout(x);
    }
    f_copy(move x); };

  () _ = = { auto x = new<int>(0);
    f_copy(move x);
    if true {
      _ = 0;
    }
    else {
      { _ = 0; }
      _ = 0;
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_copy(move x);
    }
    else {
      { _ = 0; }
      _ = 0;
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      _ = 0;
    }
    else {
      { f_copy(move x); }
      _ = 0;
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      _ = 0;
    }
    else {
      { _ = 0; }
      f_copy(move x);
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      _ = 0;
    }
    else {
      { _ = 0; }
      _ = 0;
    }
    f_copy(move x); };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_copy(move x);
    }
    else {
      { f_copy(move x); }
      _ = 0;
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_copy(move x);
    }
    else {
      { f_inout(x); }
      f_copy(move x);
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_inout(x);
    }
    else {
      { f_inout(x); }
      f_inout(x);
    }
    f_inout(x);
    if true {
      f_copy(move x);
    } };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_inout(x);
    }
    else {
      { f_inout(x); }
      f_inout(x);
    }
    if true {
      f_inout(x);
    }
    f_copy(move x); };

  () _ = = { auto x = new<int>(0);
    f_inout(x);
    if true {
      f_copy(move x);
    }
    else {
      { f_inout(x); }
      f_inout(x);
      if true {
        f_copy(move x);
      }
    }
    _ = 0; };

  () _ = = { auto x = new<int>(0);
    if true {
      if true {
        if true {
          f_copy(move x);
        }
      }
    }
    else {
    } };

  () _ = = { auto x = new<int>(0);
    if true {
      if true {
        if true {
          f_copy(move x);
        }
      }
    }
    else {
      f_copy(move x);
    } };

  () _ = = { auto x = new<int>(0);
    if true {
    }
    else {
      if true {
        if true {
          f_copy(move x);
        }
      }
    } };

  () _ = = { auto x = new<int>(0);
    if true {
      f_copy(move x);
    }
    else {
      if true {
        if true {
          f_copy(move x);
        }
      }
    } };

  () _ = = { auto x = new<int>(0);
    auto if true {
      y = new<int>(0);
      f_copy(move x);
      f_copy(move y);
    }
    else {
      if true {
        if true {
          f_inout(x);
        }
        f_copy(move x);
      }
    } };

  () _ = = { auto x = new<int>(0);
    auto if true {
      y = new<int>(0);
      if true { }
      else {
        f_copy(move x);
        f_copy(move y);
      }
    }
    auto else {
      if true {
        if true {
          y = new<int>(0);
          f_copy(move y);
          f_inout(x);
        }
        f_copy(move x);
      }
    } };

  () _ = = { auto x = new<int>(0);
    auto if true {
      y = new<int>(0);
      if true { }
      else {
        f_copy(move x);
        f_copy(move y);
      }
    }
    auto else {
      y = new<int>(0);
      if true {
        if true {
          f_copy(move x);
        }
        else {
          f_copy(move x);
        }
        f_copy(move y);
      }
    } };

  () _ = = { auto x = new<int>(0);
    if true {
      f_copy(move x);
    }
    auto else {
      if true {
        x = new<int>(0);
        if true {
          f_inout(x);
        }
        else {
        }
        f_copy(move x);
      }
      f_copy(move x);
    } };

  () _ = = { auto x = new<int>(0);
    auto if true {
      if true {
        x = new<int>(0);
        if true {
          f_inout(x);
        }
        else {
        }
        f_copy(move x);
      }
      f_copy(move x);
    }
    else {
      f_copy(move x);
    } };

  () _ = = { auto x = new<int>(0);

    if true {
      f_inout(x);
    }

    if true {
      if true {
        f_copy(move x);
      }
    } };

  () _ = = { auto x = new<int>(0);
    if true {
      if true {
        f_inout(x);
        if true {
        }
        else {
          f_copy(move x);
        }
      }
      else {
        if true {
        }
        else {
          f_inout(x);
        }
        f_copy(move x);
      }
    }
    else {
      if true {
        if true {
          f_inout(x);
          f_copy(move x);
        }
        else {
        }
      }
      else {
        if true {
        }
        else {
          f_inout(x);
        }
        if true {
          f_inout(x);
          if true {
            f_copy(move x);
          }
          else {
          }
        }
        else {
          if true {
            f_copy(move x);
          }
          else {
            f_copy(move x);
          }
        }
      }
    } }; }

void issue_888_0(std::string r, int size) { _ = r.size(); }
void issue_888_1(std::string _, std::move_only_function<(_:int)->int> size) { _ = 0.size(); }

void issue_890() { auto x = new<int>(0);
//   assert(identity_copy(x)* == 0);
  auto (x = new<int>(0)) assert(identity(x)* == 0); }

void issue_962(const ::std::string& s) { using ::std::string;
  (s)$" << std::endl std::cout << "A; }

void draw() { auto pos = 0;
  std::move_only_function<(_:int)->int> vertex = ();
  _ = (pos).vertex(); }

void enum_0() { std::string underlying_type;
    if true { }
    underlying_type = ""; }
void enum_1() { auto max_value = new<int>(0);
    std::reference_wrapper<const std::unique_ptr<int>> min_value = max_value;
    _ = max_value;

    auto // for  (0)
    // do   (copy x)
    // {
    //     v = new<int>(identity_copy(x));
    //     if pred(v, min_value) {
    //         min_value = std::ref(identity(v)); // Not using 'else' will never move 'v'.
    //     }
    //     if pred(v, max_value) {
    //         max_value = identity_copy(v);
    //     }
    // }

    auto y = new<bool>(false);
    auto while identity(y)* {
        v = new<int>(0);
        f_copy(move v);
    }

    auto z = new<bool>(false);
    auto do {
        v = new<int>(0);
        f_copy(move v);
    } while identity(z)*; }
void enum_2() { auto umax = new<int>(0);
    if pred(umax) {
    }
    else if pred(umax) {
    }
    else if pred_copy(move umax) {
    } }void union: type = {
  destroy(inout this) { }
  auto operator= = [](move this) { destroy();
    _ = this; }; }

my_string: @struct type = {
  string: std::string;std::size_t size = string.size();
}

void no_pessimizing_move(std::unique_ptr<int> = () ) -> (ret) {}

void deferred_non_copyable_0() { std::unique_ptr<int> p;
  p = ();
  f_copy(move p); }

auto deferred_non_copyable_1() { std::unique_ptr<int> p;
  p = ();
  return (move p); }

void deferred_non_copyable_2(std::unique_ptr<int> ) -> (p) { p = (); }

void loops() { () _ = = { auto x = new<int>(0);
    for (0)
    do (_)
    f_inout(x); };

  () _ = = { auto x = new<int>(0);
    for (0)
    next f_inout(x)
    do (_)
    { } };

  () _ = = { auto x = new<int>(0);
    for (0)
    do (_)
    assert(x.get()); };

  () _ = = { auto x = new<int>(0);
    if true {
      f_copy(move x);
    }
    else {
      while true {
        f_inout(x);
      }
    } }; }void captures: namespace = {

// Skip non captured name in function expression

f() { auto x = new<int>(0);
  f_copy(move x);
  auto id = :(forward x) = x;
  auto y = new<int>(0);
  assert(id(y)& == y&);
}

int x = = 0;

@struct type t = { std::unique_ptr<int> x;
  operator(): (move this) = {
    f_copy(move x);
    _ = :() = {
      // Should this move?
      // I.e., don't skip non-captured names, just rely on skipping hiding names.
      // An odr-use still requires capturing at Cpp1-time, and capturing would move.
      static_assert(std::is_same_v<decltype(x), std::unique_ptr<int>>);
      using captures::x;
      _ = identity(x);
    };
  } };

g: () = {
  _ = :() = {
    x := new<int>(0);
    f_copy(move x);
    _ = :() = std::array<int, :(x) = identity(x);(0)>()$; // Fails on Clang 12 (lambda in unevaluated context).
  };

  _ = :() = {
    x := new<int>(0);
    f_inout(x);
    _ = :() -> int = (:() = x$*)$();
  };
}

}

loops_and_captures: () = {
  _ = :() = {
    x := new<int>(0);
    f_copy(move x);
    for (:(x) -> _ = x)
    do (_)
    { }
  };

  _ = :() = {
    x := new<int>(0);
    f_copy(move x);
    for (:() -> _ = {
      using captures::x;
      return x;
    })
    do (_)
    { }
  };

//   _ = :() = {
//     x := new<int>(0);
//     for (:() x$*)
//     do (_)
//     { }
//   };
}

types: @struct type = {
  x: std::unique_ptr<int>;
//   f: (move this) = _ = :() x$*;
//   g: (move this) = {
//     for (:() x$*)
//     do (_)
//     { }
//   }
}

skip_hidden_names: () = {
  _ = :() = {
    x := new<int>(0);
    f_copy(move x);
    (copy x := new<int>(0))
      f_copy(move x);
  };

//   _ = :() = {
//     x := new<int>(0);
//     _ = :() = {
//       _ = x$;
//       x := new<int>(1);
//       _ = :() = {
//         _ = x$;
//       };
//     };
//   };

  _ = :() = {
    // x := new<int>(0);
    // f_copy(x);
    // for (0)
    // do (copy x)
    // _ = identity_copy(x);
    (copy x := new<int>(0))
      f_copy(move x);
  };

  _ = :() = {
    x := new<int>(0);
    f_inout(x);
    {
      f_copy(move x);
      using captures::x;
      f_inout(x);
    }
  };

  _ = :() = {
    x := new<int>(0);
    f_copy(move x);
    _ = :() = {
      static_assert(std::is_same_v<decltype(x), std::unique_ptr<int>>);
      using captures::x;
      f_inout(x);
    };
  };
}

main: (args) = {
  issue_683(args);
  issue_847_2(std::vector<std::unique_ptr<int>>());
  issue_847_5(args);
  issue_850();
  enum_0(); }