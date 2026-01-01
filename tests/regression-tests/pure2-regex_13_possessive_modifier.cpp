#include "cpp2util.h"

[[nodiscard]] auto create_result(std::string resultExpr, auto r) -> std::string;
[[nodiscard]] auto sanitize(std::string str) -> std::string;
auto test(M regex, std::string id, std::string regex_str, std::string str, std::string kind, std::string resultExpr, std::string resultExpected) -> void;

[[nodiscard]] auto create_result(std::string resultExpr, auto r) -> std::string {
    std::string result = "";
    auto get_next = /* expression kind 8 */;
    auto extract_group_and_advance = /* expression kind 8 */;
    auto extract_until = /* expression kind 8 */;
    auto iter = resultExpr.begin();
while (iter != resultExpr.end())     {
        auto next = get_next(iter);
if (next != iter)         {
            result ?op? std::string(iter, next);
        }
if (next != resultExpr.end())         {
if (*next == ''')             {
                next++;
if (*next == ''')                 {
                    next++;
                    result ?op? r.group(0);
                }
 else if (*next == ''' ?op? *next == ''')                 {
                    auto is_start = *next == ''';
                    next++;
if (*next == ''')                     {
                        next++;
                        auto group = extract_until(next, ''');
                        next++;
                        result ?op? r.group(group);
                    }
 else if (*next == ''')                     {
                        next++;
                        auto group = extract_group_and_advance(next);
                        next++;
if (is_start)                         {
                            result ?op? std::to_string(r.group_start(group));
                        }
 else                         {
                            result ?op? std::to_string(r.group_end(group));
                        }
                    }
 else                     {
                        result ?op? r.group(r.group_number() - 1);
                    }
                }
 else if (std::isdigit(*next))                 {
                    auto group = extract_group_and_advance(next);
                    result ?op? r.group(group);
                }
 else                 {
                    std::cerr << "Not implemented";
                }
            }
 else if (*next == ''')             {
                next++;
if (*next == ''' ?op? *next == ''')                 {
                    auto i = 0;
while (i < cpp2::unchecked_narrow(r.group_number()))                     {
                        {
                            auto pos = 0;
if (*next == ''')                             {
                                pos = r.group_start(i);
                            }
 else                             {
                                pos = r.group_end(i);
                            }
                            result ?op? std::to_string(pos);
                        }
                        
i++;                    }
                    next++;
                }
 else                 {
                    std::cerr << "Not implemented";
                }
            }
 else             {
                std::cerr << "Not implemented.";
            }
        }
        iter = next;
    }
    return result;
}

[[nodiscard]] auto sanitize(std::string str) -> std::string {
    str = cpp2::string_util::replace_all(str, "\a", "\\a");
    str = cpp2::string_util::replace_all(str, "\f", "\\f");
    str = cpp2::string_util::replace_all(str, "\x1b", "\\e");
    str = cpp2::string_util::replace_all(str, "\n", "\\n");
    str = cpp2::string_util::replace_all(str, "\r", "\\r");
    str = cpp2::string_util::replace_all(str, "\t", "\\t");
    return str;
}

template<typename M>
auto test(M regex, std::string id, std::string regex_str, std::string str, std::string kind, std::string resultExpr, std::string resultExpected) -> void {
    std::string warning = "";
if (regex.to_string() != regex_str)     {
        warning = "Warning: Parsed regex does not match.";
    }
    std::string status = "OK";
    auto r = regex.search(str);
if ("y" == kind ?op? "yM" == kind ?op? "yS" == kind ?op? "yB" == kind)     {
if (!r.matched)         {
            status = "Failure: Regex should apply.";
        }
 else         {
            auto result = create_result(resultExpr, r);
if (result != resultExpected)             {
                status = "Failure: Result is wrong. (is: (sanitize(result))$)";
            }
        }
    }
 else if ("n" == kind)     {
if (r.matched)         {
            status = "Failure: Regex should not apply. Result is '(r.group(0))$'";
        }
    }
 else     {
        status = "Unknown kind '(kind)$'";
    }
if (!warning.empty())     {
        warning ?op? " ";
    }
    std::cout << "(id)$_(kind)$: (status)$ (warning)$regex: (regex_str)$ parsed_regex: (regex.to_string())$ str: (sanitize(str))$ result_expr: (resultExpr)$ expected_results (sanitize(resultExpected))$" << std::endl;
}

struct test_tests_13_possessive_modifier {
    std::regex regex_01 = default;
    std::regex regex_02 = default;
    std::regex regex_03 = default;
    std::regex regex_04 = default;
    std::regex regex_05 = default;
    std::regex regex_06 = default;
    std::regex regex_07 = default;
    std::regex regex_08 = default;
    std::regex regex_09 = default;
    std::regex regex_10 = default;
    std::regex regex_11 = default;
    std::regex regex_12 = default;
    std::regex regex_13 = default;
    std::regex regex_14 = default;
    std::regex regex_15 = default;
    std::regex regex_16 = default;
    std::regex regex_17 = default;
    std::regex regex_18 = default;
    std::regex regex_19 = default;
    std::regex regex_20 = default;
    std::regex regex_21 = default;
    std::regex regex_22 = default;
    std::regex regex_23 = default;
    std::regex regex_24 = default;
    std::regex regex_25 = default;
    std::regex regex_26 = default;
    std::regex regex_27 = default;
    std::regex regex_28 = default;
    std::regex regex_29 = default;
    std::regex regex_30 = default;
    std::regex regex_31 = default;
    std::regex regex_32 = default;
    std::regex regex_33 = default;
    std::regex regex_34 = default;
    std::regex regex_35 = default;
    std::regex regex_36 = default;
    std::regex regex_37 = default;
    std::regex regex_38 = default;
    std::regex regex_39 = default;
    std::regex regex_40 = default;
    std::regex regex_41 = default;
    std::regex regex_42 = default;
    std::regex regex_43 = default;
    std::regex regex_44 = default;
    std::regex regex_45 = default;
    std::regex regex_46 = default;
    std::regex regex_47 = default;
    std::regex regex_48 = default;
auto run(auto this) -> void     {
        std::cout << "Running tests_13_possessive_modifier:" << std::endl;
        test(regex_01, "01", "R"(a++a)"", "aaaaa", "n", "R"(-)"", "-");
        test(regex_02, "02", "R"(a*+a)"", "aaaaa", "n", "R"(-)"", "-");
        test(regex_03, "03", "R"(a{1,5}+a)"", "aaaaa", "n", "R"(-)"", "-");
        test(regex_04, "04", "R"(a?+a)"", "ab", "n", "R"(-)"", "-");
        test(regex_05, "05", "R"(a++b)"", "aaaaab", "y", "R"($&)"", "aaaaab");
        test(regex_06, "06", "R"(a*+b)"", "aaaaab", "y", "R"($&)"", "aaaaab");
        test(regex_07, "07", "R"(a{1,5}+b)"", "aaaaab", "y", "R"($&)"", "aaaaab");
        test(regex_08, "08", "R"(a?+b)"", "ab", "y", "R"($&)"", "ab");
        test(regex_09, "09", "R"(fooa++a)"", "fooaaaaa", "n", "R"(-)"", "-");
        test(regex_10, "10", "R"(fooa*+a)"", "fooaaaaa", "n", "R"(-)"", "-");
        test(regex_11, "11", "R"(fooa{1,5}+a)"", "fooaaaaa", "n", "R"(-)"", "-");
        test(regex_12, "12", "R"(fooa?+a)"", "fooab", "n", "R"(-)"", "-");
        test(regex_13, "13", "R"(fooa++b)"", "fooaaaaab", "y", "R"($&)"", "fooaaaaab");
        test(regex_14, "14", "R"(fooa*+b)"", "fooaaaaab", "y", "R"($&)"", "fooaaaaab");
        test(regex_15, "15", "R"(fooa{1,5}+b)"", "fooaaaaab", "y", "R"($&)"", "fooaaaaab");
        test(regex_16, "16", "R"(fooa?+b)"", "fooab", "y", "R"($&)"", "fooab");
        test(regex_17, "17", "R"((aA)++(aA))"", "aAaAaAaAaA", "n", "R"(-)"", "aAaAaAaAaA");
        test(regex_18, "18", "R"((aA|bB)++(aA|bB))"", "aAaAbBaAbB", "n", "R"(-)"", "aAaAbBaAbB");
        test(regex_19, "19", "R"((aA)*+(aA))"", "aAaAaAaAaA", "n", "R"(-)"", "aAaAaAaAaA");
        test(regex_20, "20", "R"((aA|bB)*+(aA|bB))"", "aAaAbBaAaA", "n", "R"(-)"", "aAaAbBaAaA");
        test(regex_21, "21", "R"((aA){1,5}+(aA))"", "aAaAaAaAaA", "n", "R"(-)"", "aAaAaAaAaA");
        test(regex_22, "22", "R"((aA|bB){1,5}+(aA|bB))"", "aAaAbBaAaA", "n", "R"(-)"", "aAaAbBaAaA");
        test(regex_23, "23", "R"((aA)?+(aA))"", "aAb", "n", "R"(-)"", "aAb");
        test(regex_24, "24", "R"((aA|bB)?+(aA|bB))"", "bBb", "n", "R"(-)"", "bBb");
        test(regex_25, "25", "R"((aA)++b)"", "aAaAaAaAaAb", "y", "R"($&)"", "aAaAaAaAaAb");
        test(regex_26, "26", "R"((aA|bB)++b)"", "aAbBaAaAbBb", "y", "R"($&)"", "aAbBaAaAbBb");
        test(regex_27, "27", "R"((aA)*+b)"", "aAaAaAaAaAb", "y", "R"($&)"", "aAaAaAaAaAb");
        test(regex_28, "28", "R"((aA|bB)*+b)"", "bBbBbBbBbBb", "y", "R"($&)"", "bBbBbBbBbBb");
        test(regex_29, "29", "R"((aA){1,5}+b)"", "aAaAaAaAaAb", "y", "R"($&)"", "aAaAaAaAaAb");
        test(regex_30, "30", "R"((aA|bB){1,5}+b)"", "bBaAbBaAbBb", "y", "R"($&)"", "bBaAbBaAbBb");
        test(regex_31, "31", "R"((aA)?+b)"", "aAb", "y", "R"($&)"", "aAb");
        test(regex_32, "32", "R"((aA|bB)?+b)"", "bBb", "y", "R"($&)"", "bBb");
        test(regex_33, "33", "R"(foo(aA)++(aA))"", "fooaAaAaAaAaA", "n", "R"(-)"", "fooaAaAaAaAaA");
        test(regex_34, "34", "R"(foo(aA|bB)++(aA|bB))"", "foobBbBbBaAaA", "n", "R"(-)"", "foobBbBbBaAaA");
        test(regex_35, "35", "R"(foo(aA)*+(aA))"", "fooaAaAaAaAaA", "n", "R"(-)"", "fooaAaAaAaAaA");
        test(regex_36, "36", "R"(foo(aA|bB)*+(aA|bB))"", "foobBaAbBaAaA", "n", "R"(-)"", "foobBaAbBaAaA");
        test(regex_37, "37", "R"(foo(aA){1,5}+(aA))"", "fooaAaAaAaAaA", "n", "R"(-)"", "fooaAaAaAaAaA");
        test(regex_38, "38", "R"(foo(aA|bB){1,5}+(aA|bB))"", "fooaAbBbBaAaA", "n", "R"(-)"", "fooaAbBbBaAaA");
        test(regex_39, "39", "R"(foo(aA)?+(aA))"", "fooaAb", "n", "R"(-)"", "fooaAb");
        test(regex_40, "40", "R"(foo(aA|bB)?+(aA|bB))"", "foobBb", "n", "R"(-)"", "foobBb");
        test(regex_41, "41", "R"(foo(aA)++b)"", "fooaAaAaAaAaAb", "y", "R"($&)"", "fooaAaAaAaAaAb");
        test(regex_42, "42", "R"(foo(aA|bB)++b)"", "foobBaAbBaAbBb", "y", "R"($&)"", "foobBaAbBaAbBb");
        test(regex_43, "43", "R"(foo(aA)*+b)"", "fooaAaAaAaAaAb", "y", "R"($&)"", "fooaAaAaAaAaAb");
        test(regex_44, "44", "R"(foo(aA|bB)*+b)"", "foobBbBaAaAaAb", "y", "R"($&)"", "foobBbBaAaAaAb");
        test(regex_45, "45", "R"(foo(aA){1,5}+b)"", "fooaAaAaAaAaAb", "y", "R"($&)"", "fooaAaAaAaAaAb");
        test(regex_46, "46", "R"(foo(aA|bB){1,5}+b)"", "foobBaAaAaAaAb", "y", "R"($&)"", "foobBaAaAaAaAb");
        test(regex_47, "47", "R"(foo(aA)?+b)"", "fooaAb", "y", "R"($&)"", "fooaAb");
        test(regex_48, "48", "R"(foo(aA|bB)?+b)"", "foobBb", "y", "R"($&)"", "foobBb");
        std::cout << std::endl;
    }
    
    // @regex metafunction: compile-time regex validation
    // Note: regex members are compiled at construction
};

auto main() -> void {
    test_tests_13_possessive_modifier().run();
}

