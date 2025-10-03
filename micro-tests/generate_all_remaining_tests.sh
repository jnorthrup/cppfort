#!/bin/bash
# Generate remaining 400 micro tests across all categories
# This is a streamlined batch generator

cd "$(dirname "$0")"

python3 << 'PYTHON_SCRIPT'
import os

test_num = 301  # Start after cf001-cf100, ar001-ar080, mem001-mem120

def write_test(directory, filename, code, desc):
    global test_num
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/{filename}", 'w') as f:
        f.write(f"// {filename}\n// {desc}\n// Test #{test_num:03d}\n\n{code}")
    test_num += 1

# === CATEGORY 4: FUNCTIONS (100 tests) ===
# Basic functions (20)
for i in range(1, 21):
    write_test("functions", f"fn{i:03d}-test.cpp", f"""
int func{i}(int x) {{ return x + {i}; }}
int main() {{ return func{i}({i*2}); }}
""", f"Function test {i}")

# Function overloading (20)
for i in range(21, 41):
    write_test("functions", f"fn{i:03d}-overload.cpp", f"""
int func(int x) {{ return x; }}
float func(float x) {{ return x; }}
int main() {{ return (int)func({i}); }}
""", f"Function overload {i}")

# Templates (20)
for i in range(41, 61):
    write_test("functions", f"fn{i:03d}-template.cpp", f"""
template<typename T>
T func(T x) {{ return x + {i}; }}
int main() {{ return func({i}); }}
""", f"Function template {i}")

# Lambdas (20)
for i in range(61, 81):
    write_test("functions", f"fn{i:03d}-lambda.cpp", f"""
int main() {{
    auto f = [](int x) {{ return x + {i}; }};
    return f({i});
}}
""", f"Lambda {i}")

# Variadic & advanced (20)
for i in range(81, 101):
    write_test("functions", f"fn{i:03d}-variadic.cpp", f"""
template<typename... Args>
int sum(Args... args) {{ return (args + ...); }}
int main() {{ return sum(1, 2, {i}); }}
""", f"Variadic template {i}")

# === CATEGORY 5: CLASSES (120 tests) ===
# Basic classes (30)
for i in range(1, 31):
    write_test("classes", f"cls{i:03d}-basic.cpp", f"""
class Test{{ int x; public: Test(int v):x(v){{}} int get(){{return x;}} }};
int main(){{ Test t({i}); return t.get(); }}
""", f"Basic class {i}")

# Inheritance (30)
for i in range(31, 61):
    write_test("classes", f"cls{i:03d}-inherit.cpp", f"""
class Base{{ public: virtual int get(){{return {i};}} }};
class Derived: public Base{{ public: int get(){{return {i}+1;}} }};
int main(){{ Derived d; return d.get(); }}
""", f"Inheritance {i}")

# Constructors/destructors (30)
for i in range(61, 91):
    write_test("classes", f"cls{i:03d}-ctor.cpp", f"""
class Test{{ int x; public: Test():x({i}){{}} ~Test(){{}} int get(){{return x;}} }};
int main(){{ Test t; return t.get(); }}
""", f"Constructor/destructor {i}")

# Operators (30)
for i in range(91, 121):
    write_test("classes", f"cls{i:03d}-operator.cpp", f"""
class Test{{ int x; public: Test(int v):x(v){{}} Test operator+(const Test& o){{return Test(x+o.x);}} int get(){{return x;}} }};
int main(){{ Test a({i}), b(1); return (a+b).get(); }}
""", f"Operator overload {i}")

# === CATEGORY 6: TEMPLATES (80 tests) ===
for i in range(1, 81):
    write_test("templates", f"tpl{i:03d}-test.cpp", f"""
template<typename T> T add(T a, T b) {{ return a + b; }}
int main() {{ return add({i}, 1); }}
""", f"Template test {i}")

# === CATEGORY 7: STDLIB (60 tests) ===
for i in range(1, 61):
    write_test("stdlib", f"std{i:03d}-test.cpp", f"""
#include <vector>
int main() {{ std::vector<int> v{{1,2,{i}}}; return v[2]; }}
""", f"Standard library test {i}")

# === CATEGORY 8: EXCEPTIONS (40 tests) ===
for i in range(1, 41):
    write_test("exceptions", f"exc{i:03d}-test.cpp", f"""
int func() {{ try {{ throw {i}; }} catch(int e) {{ return e; }} return 0; }}
int main() {{ return func(); }}
""", f"Exception test {i}")

# === CATEGORY 9: MODERN C++ (60 tests) ===
for i in range(1, 61):
    write_test("modern-cpp", f"mod{i:03d}-test.cpp", f"""
#include <utility>
auto func() {{ return std::make_pair({i}, {i}+1); }}
int main() {{ auto [a,b] = func(); return a; }}
""", f"Modern C++ test {i}")

# === CATEGORY 10: EDGE CASES (40 tests) ===
for i in range(1, 41):
    write_test("edge-cases", f"edge{i:03d}-test.cpp", f"""
int func() {{ volatile int x = {i}; return x; }}
int main() {{ return func(); }}
""", f"Edge case test {i}")

print(f"Generated {test_num-301} additional tests!")
print(f"Total tests across all categories: {test_num-1}")
PYTHON_SCRIPT

echo "Test generation complete!"
