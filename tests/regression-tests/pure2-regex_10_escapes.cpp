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

struct test_tests_10_escapes {
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
auto run(auto this) -> void     {
        std::cout << "Running tests_10_escapes:" << std::endl;
        test(regex_01, "01", "R"(a\(b)"", "a(b", "y", "R"($&-$1)"", "a(b-");
        test(regex_02, "02", "R"(a\(*b)"", "ab", "y", "R"($&)"", "ab");
        test(regex_03, "03", "R"(a\(*b)"", "a((b", "y", "R"($&)"", "a((b");
        test(regex_04, "04", "R"(a\\b)"", "a\\b", "y", "R"($&)"", "a\\b");
        test(regex_05, "05", "R"(foo(\h+)bar)"", "foo\tbar", "y", "R"($1)"", "\t");
        test(regex_06, "06", "R"((\H+)(\h))"", "foo\tbar", "y", "R"($1-$2)"", "foo-\t");
        test(regex_07, "07", "R"((\h+)(\H))"", "foo\tbar", "y", "R"($1-$2)"", "\t-b");
        test(regex_08, "08", "R"(foo(\h)bar)"", "foo\tbar", "y", "R"($1)"", "\t");
        test(regex_09, "09", "R"((\H)(\h))"", "foo\tbar", "y", "R"($1-$2)"", "o-\t");
        test(regex_10, "10", "R"((\h)(\H))"", "foo\tbar", "y", "R"($1-$2)"", "\t-b");
        test(regex_11, "11", "R"(foo(\v+)bar)"", "foo\r\n\r\n\nbar", "y", "R"($1)"", "\r\n\r\n\n");
        test(regex_12, "12", "R"((\V+)(\v))"", "foo\r\n\r\n\nbar", "y", "R"($1-$2)"", "foo-\r");
        test(regex_13, "13", "R"((\v+)(\V))"", "foo\r\n\r\n\nbar", "y", "R"($1-$2)"", "\r\n\r\n\n-b");
        test(regex_14, "14", "R"(foo(\v)bar)"", "foo\rbar", "y", "R"($1)"", "\r");
        test(regex_15, "15", "R"((\V)(\v))"", "foo\rbar", "y", "R"($1-$2)"", "o-\r");
        test(regex_16, "16", "R"((\v)(\V))"", "foo\rbar", "y", "R"($1-$2)"", "\r-b");
        test(regex_17, "17", "R"(foo\t\n\r\f\a\ebar)"", "foo\t\n\r\f\a");
        test(regex_18, "18", "R"(foo\Kbar)"", "foobar", "y", "R"($&)"", "bar");
        test(regex_19, "19", "R"(\x41\x42)"", "AB", "y", "R"($&)"", "AB");
        test(regex_20, "20", "R"(\101\o{102})"", "AB", "y", "R"($&)"", "AB");
        std::cout << std::endl;
    }
    
    // @regex metafunction: compile-time regex validation
    // Note: regex members are compiled at construction
};

auto main() -> void {
    test_tests_10_escapes().run();
}

