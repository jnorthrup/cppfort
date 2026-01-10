

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================

// Combinator Corpus Tests
// Tests combinator usage patterns that should compile and run correctly
// These tests exercise the transpiler's handling of combinator expressions

[[nodiscard]] auto combinator_corpus_basic() -> int;

[[nodiscard]] auto combinator_corpus_take() -> std::string;

[[nodiscard]] auto combinator_corpus_skip() -> std::string;

[[nodiscard]] auto combinator_corpus_pipeline() -> int;

[[nodiscard]] auto combinator_corpus_map() -> int;

[[nodiscard]] auto combinator_corpus_fold() -> int;

[[nodiscard]] auto combinator_corpus_split() -> int;

[[nodiscard]] auto combinator_corpus_enumerate() -> int;

[[nodiscard]] auto combinator_corpus_chain() -> std::string;

[[nodiscard]] auto combinator_corpus_find() -> char;

[[nodiscard]] auto combinator_corpus_any_all() -> bool;

// Test entry point that validates all corpus tests
[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto combinator_corpus_basic() -> int{
    // Basic ByteBuffer creation and iteration
    std::string data {"Hello"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    auto count {0}; 
    for ( auto const& c : cpp2::move(buf) ) {
        ++count;
    }
    return count;  // Should return 5
}

[[nodiscard]] auto combinator_corpus_take() -> std::string{
    std::string data {"0123456789"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    std::string result {""}; 
    auto taken {cpp2::combinators::take(cpp2::move(buf), 5)}; 
    for ( auto const& c : cpp2::move(taken) ) {
        result += c;
    }
    return result;  // Should return "01234"
}

[[nodiscard]] auto combinator_corpus_skip() -> std::string{
    std::string data {"0123456789"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    std::string result {""}; 
    auto skipped {cpp2::combinators::skip(cpp2::move(buf), 5)}; 
    for ( auto const& c : cpp2::move(skipped) ) {
        result += c;
    }
    return result;  // Should return "56789"
}

[[nodiscard]] auto combinator_corpus_pipeline() -> int{
    std::string data {"aAbBcCdDeE"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    // Pipeline: skip 2, take 6, filter uppercase
    auto pipeline {cpp2::combinators::from(cpp2::move(buf)) 
        | cpp2::combinators::curried::skip(2) 
        | cpp2::combinators::curried::take(6) 
        | cpp2::combinators::curried::filter([](cpp2::impl::in<char> c) -> bool { return std::isupper(c) != 0;  })}; 

    auto count {0}; 
    for ( auto const& c : cpp2::move(pipeline) ) {
        ++count;
    }
    return count;  // Should return 3 (B, C, D)
}

[[nodiscard]] auto combinator_corpus_map() -> int{
    std::string data {"abc"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    // Map to ASCII values and sum
    auto mapped {cpp2::combinators::from(cpp2::move(buf)) 
        | cpp2::combinators::curried::map([](cpp2::impl::in<char> c) -> int { return cpp2::impl::as_<int>(c);  })}; 

    auto sum {0}; 
    for ( auto const& v : cpp2::move(mapped) ) {
        sum += v;
    }
    return sum;  // 97+98+99 = 294
}

[[nodiscard]] auto combinator_corpus_fold() -> int{
    std::string data {"12345"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    return CPP2_UFCS(fold)(cpp2::combinators::reduce_from(cpp2::move(buf))
        , 0, [](cpp2::impl::in<int> acc, cpp2::impl::in<char> c) -> int { return acc + (c - '0');  }); 
    // Should return 15
}

[[nodiscard]] auto combinator_corpus_split() -> int{
    std::string data {"a,b,c,d,e"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    auto parts {cpp2::combinators::split(cpp2::move(buf), ',')}; 
    auto count {0}; 
    for ( auto const& part : cpp2::move(parts) ) {
        ++count;
    }
    return count;  // Should return 5
}

[[nodiscard]] auto combinator_corpus_enumerate() -> int{
    std::string data {"abc"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    auto enumerated {cpp2::combinators::from(cpp2::move(buf)) 
        | cpp2::combinators::curried::enumerate()}; 

    auto sum {0}; 
    for ( auto const& pair : cpp2::move(enumerated) ) {
        sum += pair.first;
    }
    return sum;  // 0+1+2 = 3
}

[[nodiscard]] auto combinator_corpus_chain() -> std::string{
    std::string data {"The Quick Brown Fox"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    // Complex chain: filter non-space, map to upper, take 10
    std::string result {""}; 

    auto chain {cpp2::combinators::from(cpp2::move(buf)) 
        | cpp2::combinators::curried::filter([](cpp2::impl::in<char> c) -> bool { return c != ' ';  }) 
        | cpp2::combinators::curried::map([](cpp2::impl::in<char> c) -> char { return cpp2::impl::as_<char>(std::toupper(c));  }) 
        | cpp2::combinators::curried::take(10)}; 

    for ( auto const& c : cpp2::move(chain) ) {
        result += c;
    }
    return result;  // "THEQUICKBR"
}

[[nodiscard]] auto combinator_corpus_find() -> char{
    std::string data {"abcdef"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    auto found {CPP2_UFCS(find)(cpp2::combinators::reduce_from(cpp2::move(buf))
        , [](cpp2::impl::in<char> c) -> bool { return c == 'd';  })}; 

    if (CPP2_UFCS(has_value)(found)) {
        return CPP2_UFCS(value)(cpp2::move(found)); 
    }
    return '?'; 
}

[[nodiscard]] auto combinator_corpus_any_all() -> bool{
    std::string data {"12345"}; 
    cpp2::ByteBuffer buf {CPP2_UFCS(data)(data), CPP2_UFCS(size)(cpp2::move(data))}; 

    auto all_digits {CPP2_UFCS(all)(cpp2::combinators::reduce_from(cpp2::move(buf))
        , [](cpp2::impl::in<char> c) -> bool { return std::isdigit(c) != 0;  })}; 

    return all_digits;  // true
}

[[nodiscard]] auto main() -> int{
    std::cout << "=== Combinator Corpus Tests ===\n";

    auto failures {0}; 

    // Test basic
    if (combinator_corpus_basic() != 5) {
        std::cout << "FAIL: combinator_corpus_basic\n";
        ++failures;
    }

    // Test take
    if (combinator_corpus_take() != "01234") {
        std::cout << "FAIL: combinator_corpus_take\n";
        ++failures;
    }

    // Test skip
    if (combinator_corpus_skip() != "56789") {
        std::cout << "FAIL: combinator_corpus_skip\n";
        ++failures;
    }

    // Test pipeline
    if (combinator_corpus_pipeline() != 3) {
        std::cout << "FAIL: combinator_corpus_pipeline\n";
        ++failures;
    }

    // Test map
    if (combinator_corpus_map() != 294) {
        std::cout << "FAIL: combinator_corpus_map\n";
        ++failures;
    }

    // Test fold
    if (combinator_corpus_fold() != 15) {
        std::cout << "FAIL: combinator_corpus_fold\n";
        ++failures;
    }

    // Test split
    if (combinator_corpus_split() != 5) {
        std::cout << "FAIL: combinator_corpus_split\n";
        ++failures;
    }

    // Test enumerate
    if (combinator_corpus_enumerate() != 3) {
        std::cout << "FAIL: combinator_corpus_enumerate\n";
        ++failures;
    }

    // Test chain
    if (combinator_corpus_chain() != "THEQUICKBR") {
        std::cout << "FAIL: combinator_corpus_chain\n";
        ++failures;
    }

    // Test find
    if (combinator_corpus_find() != 'd') {
        std::cout << "FAIL: combinator_corpus_find\n";
        ++failures;
    }

    // Test any/all
    if (!(combinator_corpus_any_all())) {
        std::cout << "FAIL: combinator_corpus_any_all\n";
        ++failures;
    }

    if (failures == 0) {
        std::cout << "All 11 corpus tests PASSED\n";
        return 0; 
    }

    std::cout << cpp2::move(failures) << " corpus tests FAILED\n";
    return 1; 
}

