//  cpp2_combinators.hpp - Convenience header for all cppfort combinators
//
//  Copyright 2024-2026 cppfort project
//  SPDX-License-Identifier: Apache-2.0
//
//  This header provides all combinator types and functions in one place.
//  Include this instead of individual combinator headers.

#ifndef CPP2_COMBINATORS_HPP
#define CPP2_COMBINATORS_HPP

// Core types
#include "bytebuffer.hpp"
#include "strview.hpp"
#include "lazy_iterator.hpp"

// All combinator categories
#include "combinators/structural.hpp"      // take, skip, slice, split, chunk, window
#include "combinators/transformation.hpp"  // map, filter, flat_map, enumerate, zip, intersperse
#include "combinators/reduction.hpp"       // fold, reduce, scan, find, all/any/count
#include "combinators/parsing.hpp"         // byte, bytes, until, while_pred, endian parsers

#endif // CPP2_COMBINATORS_HPP
