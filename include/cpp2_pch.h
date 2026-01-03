//  cpp2_pch.h - Precompiled header for cppfort-generated code
//
//  Usage:
//    clang++ -std=c++20 -x c++-header cpp2_pch.h -o cpp2_pch.h.pch
//    clang++ -std=c++20 -include-pch cpp2_pch.h.pch mycode.cpp
//
//  Or with GCC:
//    g++ -std=c++20 -x c++-header cpp2_pch.h -o cpp2_pch.h.gch
//    g++ -std=c++20 mycode.cpp  (auto-finds .gch in same dir)

#ifndef CPP2_PCH_H
#define CPP2_PCH_H

// Heavy standard library headers - prime PCH candidates
#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <set>
#include <memory>
#include <optional>
#include <variant>
#include <expected>
#include <functional>
#include <utility>
#include <tuple>
#include <type_traits>
#include <concepts>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <ranges>
#include <span>
#include <format>
#include <source_location>

// The actual runtime
#include "cpp2_runtime.h"

// ============================================================================
// Combinator library (ByteBuffer, StrView, parsing/transformation)
// ============================================================================

#include "bytebuffer.hpp"
#include "strview.hpp"
#include "lazy_iterator.hpp"
#include "combinators/structural.hpp"
#include "combinators/transformation.hpp"
#include "combinators/reduction.hpp"
#include "combinators/parsing.hpp"

#endif // CPP2_PCH_H
