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

struct test_tests_12_case_insensitive {
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
    std::regex regex_49 = default;
    std::regex regex_50 = default;
    std::regex regex_51 = default;
    std::regex regex_52 = default;
    std::regex regex_53 = default;
    std::regex regex_54 = default;
    std::regex regex_55 = default;
    std::regex regex_56 = default;
    std::regex regex_57 = default;
    std::regex regex_58 = default;
    std::regex regex_59 = default;
    std::regex regex_60 = default;
    std::regex regex_61 = default;
    std::regex regex_62 = default;
    std::regex regex_63 = default;
    std::regex regex_64 = default;
    std::regex regex_65 = default;
    std::regex regex_66 = default;
    std::regex regex_67 = default;
    std::regex regex_68 = default;
    std::regex regex_69 = default;
    std::regex regex_70 = default;
    std::regex regex_71 = default;
    std::regex regex_72 = default;
    std::regex regex_73 = default;
    std::regex regex_74 = default;
    std::regex regex_75 = default;
    std::regex regex_76 = default;
    std::regex regex_77 = default;
    std::regex regex_78 = default;
    std::regex regex_79 = default;
    std::regex regex_80 = default;
    std::regex regex_81 = default;
    std::regex regex_82 = default;
    std::regex regex_83 = default;
    std::regex regex_84 = default;
    std::regex regex_85 = default;
    std::regex regex_86 = default;
    std::regex regex_87 = default;
    std::regex regex_88 = default;
    std::regex regex_89 = default;
    std::regex regex_90 = default;
    std::regex regex_91 = default;
    std::regex regex_92 = default;
    std::regex regex_93 = default;
    std::regex regex_94 = default;
    std::regex regex_95 = default;
    std::regex regex_96 = default;
    std::regex regex_97 = default;
    std::regex regex_98 = default;
    std::regex regex_99 = default;
    std::regex regex_100 = default;
    std::regex regex_101 = default;
    std::regex regex_102 = default;
    std::regex regex_103 = default;
    std::regex regex_104 = default;
    std::regex regex_105 = default;
    std::regex regex_106 = default;
    std::regex regex_107 = default;
    std::regex regex_108 = default;
    std::regex regex_109 = default;
    std::regex regex_110 = default;
    std::regex regex_111 = default;
    std::regex regex_112 = default;
    std::regex regex_113 = default;
    std::regex regex_114 = default;
    std::regex regex_115 = default;
    std::regex regex_116 = default;
    std::regex regex_117 = default;
    std::regex regex_118 = default;
    std::regex regex_119 = default;
auto run(auto this) -> void     {
        std::cout << "Running tests_12_case_insensitive:" << std::endl;
        test(regex_01, "01", "R"('abc'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_02, "02", "R"('abc'i)"", "XBC", "n", "R"(-)"", "-");
        test(regex_03, "03", "R"('abc'i)"", "AXC", "n", "R"(-)"", "-");
        test(regex_04, "04", "R"('abc'i)"", "ABX", "n", "R"(-)"", "-");
        test(regex_05, "05", "R"('abc'i)"", "XABCY", "y", "R"($&)"", "ABC");
        test(regex_06, "06", "R"('abc'i)"", "ABABC", "y", "R"($&)"", "ABC");
        test(regex_07, "07", "R"('ab*c'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_08, "08", "R"('ab*bc'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_09, "09", "R"('ab*bc'i)"", "ABBC", "y", "R"($&)"", "ABBC");
        test(regex_10, "10", "R"('ab*?bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_11, "11", "R"('ab{0,}?bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_12, "12", "R"('ab+?bc'i)"", "ABBC", "y", "R"($&)"", "ABBC");
        test(regex_13, "13", "R"('ab+bc'i)"", "ABC", "n", "R"(-)"", "-");
        test(regex_14, "14", "R"('ab+bc'i)"", "ABQ", "n", "R"(-)"", "-");
        test(regex_15, "15", "R"('ab{1,}bc'i)"", "ABQ", "n", "R"(-)"", "-");
        test(regex_16, "16", "R"('ab+bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_17, "17", "R"('ab{1,}?bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_18, "18", "R"('ab{1,3}?bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_19, "19", "R"('ab{3,4}?bc'i)"", "ABBBBC", "y", "R"($&)"", "ABBBBC");
        test(regex_20, "20", "R"('ab{4,5}?bc'i)"", "ABBBBC", "n", "R"(-)"", "-");
        test(regex_21, "21", "R"('ab??bc'i)"", "ABBC", "y", "R"($&)"", "ABBC");
        test(regex_22, "22", "R"('ab??bc'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_23, "23", "R"('ab{0,1}?bc'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_24, "24", "R"('ab??bc'i)"", "ABBBBC", "n", "R"(-)"", "-");
        test(regex_25, "25", "R"('ab??c'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_26, "26", "R"('ab{0,1}?c'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_27, "27", "R"('^abc$'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_28, "28", "R"('^abc$'i)"", "ABCC", "n", "R"(-)"", "-");
        test(regex_29, "29", "R"('^abc'i)"", "ABCC", "y", "R"($&)"", "ABC");
        test(regex_30, "30", "R"('^abc$'i)"", "AABC", "n", "R"(-)"", "-");
        test(regex_31, "31", "R"('abc$'i)"", "AABC", "y", "R"($&)"", "ABC");
        test(regex_32, "32", "R"('^'i)"", "ABC", "y", "R"($&)"", "");
        test(regex_33, "33", "R"('$'i)"", "ABC", "y", "R"($&)"", "");
        test(regex_34, "34", "R"('a.c'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_35, "35", "R"('a.c'i)"", "AXC", "y", "R"($&)"", "AXC");
        test(regex_36, "36", "R"('a\Nc'i)"", "ABC", "y", "R"($&)"", "ABC");
        test(regex_37, "37", "R"('a.*?c'i)"", "AXYZC", "y", "R"($&)"", "AXYZC");
        test(regex_38, "38", "R"('a.*c'i)"", "AXYZD", "n", "R"(-)"", "-");
        test(regex_39, "39", "R"('a[bc]d'i)"", "ABC", "n", "R"(-)"", "-");
        test(regex_40, "40", "R"('a[bc]d'i)"", "ABD", "y", "R"($&)"", "ABD");
        test(regex_41, "41", "R"('a[b-d]e'i)"", "ABD", "n", "R"(-)"", "-");
        test(regex_42, "42", "R"('a[b-d]e'i)"", "ACE", "y", "R"($&)"", "ACE");
        test(regex_43, "43", "R"('a[b-d]'i)"", "AAC", "y", "R"($&)"", "AC");
        test(regex_44, "44", "R"('a[-b]'i)"", "A-", "y", "R"($&)"", "A-");
        test(regex_45, "45", "R"('a[b-]'i)"", "A-", "y", "R"($&)"", "A-");
        test(regex_46, "46", "R"('a]'i)"", "A]", "y", "R"($&)"", "A]");
        test(regex_47, "47", "R"('a[]]b'i)"", "A]B", "y", "R"($&)"", "A]B");
        test(regex_48, "48", "R"('a[^bc]d'i)"", "AED", "y", "R"($&)"", "AED");
        test(regex_49, "49", "R"('a[^bc]d'i)"", "ABD", "n", "R"(-)"", "-");
        test(regex_50, "50", "R"('a[^-b]c'i)"", "ADC", "y", "R"($&)"", "ADC");
        test(regex_51, "51", "R"('a[^-b]c'i)"", "A-C", "n", "R"(-)"", "-");
        test(regex_52, "52", "R"('a[^]b]c'i)"", "A]C", "n", "R"(-)"", "-");
        test(regex_53, "53", "R"('a[^]b]c'i)"", "ADC", "y", "R"($&)"", "ADC");
        test(regex_54, "54", "R"('ab|cd'i)"", "ABC", "y", "R"($&)"", "AB");
        test(regex_55, "55", "R"('ab|cd'i)"", "ABCD", "y", "R"($&)"", "AB");
        test(regex_56, "56", "R"('()ef'i)"", "DEF", "y", "R"($&-$1)"", "EF-");
        test(regex_57, "57", "R"('$b'i)"", "B", "n", "R"(-)"", "-");
        test(regex_58, "58", "R"('a\(b'i)"", "A(B", "y", "R"($&-$1)"", "A(B-");
        test(regex_59, "59", "R"('a\(*b'i)"", "AB", "y", "R"($&)"", "AB");
        test(regex_60, "60", "R"('a\(*b'i)"", "A((B", "y", "R"($&)"", "A((B");
        test(regex_61, "61", "R"('a\\b'i)"", "A\\B", "y", "R"($&)"", "A\\B");
        test(regex_62, "62", "R"('((a))'i)"", "ABC", "y", "R"($&-$1-$2)"", "A-A-A");
        test(regex_63, "63", "R"('(a)b(c)'i)"", "ABC", "y", "R"($&-$1-$2)"", "ABC-A-C");
        test(regex_64, "64", "R"('a+b+c'i)"", "AABBABC", "y", "R"($&)"", "ABC");
        test(regex_65, "65", "R"('a{1,}b{1,}c'i)"", "AABBABC", "y", "R"($&)"", "ABC");
        test(regex_66, "66", "R"('a.+?c'i)"", "ABCABC", "y", "R"($&)"", "ABC");
        test(regex_67, "67", "R"('a.*?c'i)"", "ABCABC", "y", "R"($&)"", "ABC");
        test(regex_68, "68", "R"('a.{0,5}?c'i)"", "ABCABC", "y", "R"($&)"", "ABC");
        test(regex_69, "69", "R"('(a+|b)*'i)"", "AB", "y", "R"($&-$1)"", "AB-B");
        test(regex_70, "70", "R"('(a+|b){0,}'i)"", "AB", "y", "R"($&-$1)"", "AB-B");
        test(regex_71, "71", "R"('(a+|b)+'i)"", "AB", "y", "R"($&-$1)"", "AB-B");
        test(regex_72, "72", "R"('(a+|b){1,}'i)"", "AB", "y", "R"($&-$1)"", "AB-B");
        test(regex_73, "73", "R"('(a+|b)?'i)"", "AB", "y", "R"($&-$1)"", "A-A");
        test(regex_74, "74", "R"('(a+|b){0,1}'i)"", "AB", "y", "R"($&-$1)"", "A-A");
        test(regex_75, "75", "R"('(a+|b){0,1}?'i)"", "AB", "y", "R"($&-$1)"", "-");
        test(regex_76, "76", "R"('[^ab]*'i)"", "CDE", "y", "R"($&)"", "CDE");
        test(regex_77, "77", "R"('abc'i)"", "", "n", "R"(-)"", "-");
        test(regex_78, "78", "R"('a*'i)"", "", "y", "R"($&)"", "");
        test(regex_79, "79", "R"('([abc])*d'i)"", "ABBBCD", "y", "R"($&-$1)"", "ABBBCD-C");
        test(regex_80, "80", "R"('([abc])*bcd'i)"", "ABCD", "y", "R"($&-$1)"", "ABCD-A");
        test(regex_81, "81", "R"('a|b|c|d|e'i)"", "E", "y", "R"($&)"", "E");
        test(regex_82, "82", "R"('(a|b|c|d|e)f'i)"", "EF", "y", "R"($&-$1)"", "EF-E");
        test(regex_83, "83", "R"('abcd*efg'i)"", "ABCDEFG", "y", "R"($&)"", "ABCDEFG");
        test(regex_84, "84", "R"('ab*'i)"", "XABYABBBZ", "y", "R"($&)"", "AB");
        test(regex_85, "85", "R"('ab*'i)"", "XAYABBBZ", "y", "R"($&)"", "A");
        test(regex_86, "86", "R"('(ab|cd)e'i)"", "ABCDE", "y", "R"($&-$1)"", "CDE-CD");
        test(regex_87, "87", "R"('[abhgefdc]ij'i)"", "HIJ", "y", "R"($&)"", "HIJ");
        test(regex_88, "88", "R"('^(ab|cd)e'i)"", "ABCDE", "n", "R"(x$1y)"", "XY");
        test(regex_89, "89", "R"('(abc|)ef'i)"", "ABCDEF", "y", "R"($&-$1)"", "EF-");
        test(regex_90, "90", "R"('(a|b)c*d'i)"", "ABCD", "y", "R"($&-$1)"", "BCD-B");
        test(regex_91, "91", "R"('(ab|ab*)bc'i)"", "ABC", "y", "R"($&-$1)"", "ABC-A");
        test(regex_92, "92", "R"('a([bc]*)c*'i)"", "ABC", "y", "R"($&-$1)"", "ABC-BC");
        test(regex_93, "93", "R"('a([bc]*)(c*d)'i)"", "ABCD", "y", "R"($&-$1-$2)"", "ABCD-BC-D");
        test(regex_94, "94", "R"('a([bc]+)(c*d)'i)"", "ABCD", "y", "R"($&-$1-$2)"", "ABCD-BC-D");
        test(regex_95, "95", "R"('a([bc]*)(c+d)'i)"", "ABCD", "y", "R"($&-$1-$2)"", "ABCD-B-CD");
        test(regex_96, "96", "R"('a[bcd]*dcdcde'i)"", "ADCDCDE", "y", "R"($&)"", "ADCDCDE");
        test(regex_97, "97", "R"('a[bcd]+dcdcde'i)"", "ADCDCDE", "n", "R"(-)"", "-");
        test(regex_98, "98", "R"('(ab|a)b*c'i)"", "ABC", "y", "R"($&-$1)"", "ABC-AB");
        test(regex_99, "99", "R"('((a)(b)c)(d)'i)"", "ABCD", "y", "R"($1-$2-$3-$4)"", "ABC-A-B-D");
        test(regex_100, "100", "R"('[a-zA-Z_][a-zA-Z0-9_]*'i)"", "ALPHA", "y", "R"($&)"", "ALPHA");
        test(regex_101, "101", "R"('^a(bc+|b[eh])g|.h$'i)"", "ABH", "y", "R"($&-$1)"", "BH-");
        test(regex_102, "102", "R"('(bc+d$|ef*g.|h?i(j|k))'i)"", "EFFGZ", "y", "R"($&-$1-$2)"", "EFFGZ-EFFGZ-");
        test(regex_103, "103", "R"('(bc+d$|ef*g.|h?i(j|k))'i)"", "IJ", "y", "R"($&-$1-$2)"", "IJ-IJ-J");
        test(regex_104, "104", "R"('(bc+d$|ef*g.|h?i(j|k))'i)"", "EFFG", "n", "R"(-)"", "-");
        test(regex_105, "105", "R"('(bc+d$|ef*g.|h?i(j|k))'i)"", "BCDD", "n", "R"(-)"", "-");
        test(regex_106, "106", "R"('(bc+d$|ef*g.|h?i(j|k))'i)"", "REFFGZ", "y", "R"($&-$1-$2)"", "EFFGZ-EFFGZ-");
        test(regex_107, "107", "R"('((((((((((a))))))))))'i)"", "A", "y", "R"($10)"", "A");
        test(regex_108, "108", "R"('((((((((((a))))))))))\10'i)"", "AA", "y", "R"($&)"", "AA");
        test(regex_109, "109", "R"('(((((((((a)))))))))'i)"", "A", "y", "R"($&)"", "A");
        test(regex_110, "110", "R"('multiple words of text'i)"", "UH-UH", "n", "R"(-)"", "-");
        test(regex_111, "111", "R"('multiple words'i)"", "MULTIPLE WORDS, YEAH", "y", "R"($&)"", "MULTIPLE WORDS");
        test(regex_112, "112", "R"('(.*)c(.*)'i)"", "ABCDE", "y", "R"($&-$1-$2)"", "ABCDE-AB-DE");
        test(regex_113, "113", "R"('\((.*), (.*)\)'i)"", "(A, B)", "y", "R"(($2, $1))"", "(B, A)");
        test(regex_114, "114", "R"('[k]'i)"", "AB", "n", "R"(-)"", "-");
        test(regex_115, "115", "R"('abcd'i)"", "ABCD", "y", "R"($&)"", "ABCD");
        test(regex_116, "116", "R"('a(bc)d'i)"", "ABCD", "y", "R"($1)"", "BC");
        test(regex_117, "117", "R"('a[-]?c'i)"", "AC", "y", "R"($&)"", "AC");
        test(regex_118, "118", "R"('(abc)\1'i)"", "ABCABC", "y", "R"($1)"", "ABC");
        test(regex_119, "119", "R"('([a-c]*)\1'i)"", "ABCABC", "y", "R"($1)"", "ABC");
        std::cout << std::endl;
    }
    
    // @regex metafunction: compile-time regex validation
    // Note: regex members are compiled at construction
};

auto main() -> void {
    test_tests_12_case_insensitive().run();
}

