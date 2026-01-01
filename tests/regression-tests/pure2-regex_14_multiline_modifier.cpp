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

struct test_tests_14_multiline_modifier {
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
    std::regex regex_120 = default;
    std::regex regex_121 = default;
    std::regex regex_122 = default;
    std::regex regex_123 = default;
    std::regex regex_124 = default;
    std::regex regex_125 = default;
    std::regex regex_126 = default;
    std::regex regex_127 = default;
    std::regex regex_128 = default;
    std::regex regex_129 = default;
    std::regex regex_130 = default;
    std::regex regex_131 = default;
    std::regex regex_132 = default;
    std::regex regex_133 = default;
    std::regex regex_134 = default;
    std::regex regex_135 = default;
    std::regex regex_136 = default;
    std::regex regex_137 = default;
    std::regex regex_138 = default;
    std::regex regex_139 = default;
    std::regex regex_140 = default;
    std::regex regex_141 = default;
    std::regex regex_142 = default;
    std::regex regex_143 = default;
    std::regex regex_144 = default;
    std::regex regex_145 = default;
    std::regex regex_146 = default;
    std::regex regex_147 = default;
    std::regex regex_148 = default;
    std::regex regex_149 = default;
    std::regex regex_150 = default;
    std::regex regex_151 = default;
    std::regex regex_152 = default;
    std::regex regex_153 = default;
    std::regex regex_154 = default;
    std::regex regex_155 = default;
    std::regex regex_156 = default;
    std::regex regex_157 = default;
    std::regex regex_158 = default;
    std::regex regex_159 = default;
    std::regex regex_160 = default;
    std::regex regex_161 = default;
    std::regex regex_162 = default;
    std::regex regex_163 = default;
    std::regex regex_164 = default;
    std::regex regex_165 = default;
    std::regex regex_166 = default;
    std::regex regex_167 = default;
    std::regex regex_168 = default;
    std::regex regex_169 = default;
    std::regex regex_170 = default;
    std::regex regex_171 = default;
    std::regex regex_172 = default;
    std::regex regex_173 = default;
    std::regex regex_174 = default;
    std::regex regex_175 = default;
    std::regex regex_176 = default;
    std::regex regex_177 = default;
    std::regex regex_178 = default;
    std::regex regex_179 = default;
    std::regex regex_180 = default;
    std::regex regex_181 = default;
    std::regex regex_182 = default;
    std::regex regex_183 = default;
    std::regex regex_184 = default;
    std::regex regex_185 = default;
    std::regex regex_186 = default;
    std::regex regex_187 = default;
    std::regex regex_188 = default;
    std::regex regex_189 = default;
    std::regex regex_190 = default;
    std::regex regex_191 = default;
    std::regex regex_192 = default;
    std::regex regex_193 = default;
    std::regex regex_194 = default;
    std::regex regex_195 = default;
    std::regex regex_196 = default;
    std::regex regex_197 = default;
    std::regex regex_198 = default;
    std::regex regex_199 = default;
auto run(auto this) -> void     {
        std::cout << "Running tests_14_multiline_modifier:" << std::endl;
        test(regex_01, "01", "R"(\Z)"", "a\nb\n", "y", "R"($-[0])"", "3");
        test(regex_02, "02", "R"(\z)"", "a\nb\n", "y", "R"($-[0])"", "4");
        test(regex_03, "03", "R"($)"", "a\nb\n", "y", "R"($-[0])"", "3");
        test(regex_04, "04", "R"(\Z)"", "b\na\n", "y", "R"($-[0])"", "3");
        test(regex_05, "05", "R"(\z)"", "b\na\n", "y", "R"($-[0])"", "4");
        test(regex_06, "06", "R"($)"", "b\na\n", "y", "R"($-[0])"", "3");
        test(regex_07, "07", "R"(\Z)"", "b\na", "y", "R"($-[0])"", "3");
        test(regex_08, "08", "R"(\z)"", "b\na", "y", "R"($-[0])"", "3");
        test(regex_09, "09", "R"($)"", "b\na", "y", "R"($-[0])"", "3");
        test(regex_10, "10", "R"('\Z'm)"", "a\nb\n", "y", "R"($-[0])"", "3");
        test(regex_11, "11", "R"('\z'm)"", "a\nb\n", "y", "R"($-[0])"", "4");
        test(regex_12, "12", "R"('$'m)"", "a\nb\n", "y", "R"($-[0])"", "1");
        test(regex_13, "13", "R"('\Z'm)"", "b\na\n", "y", "R"($-[0])"", "3");
        test(regex_14, "14", "R"('\z'm)"", "b\na\n", "y", "R"($-[0])"", "4");
        test(regex_15, "15", "R"('$'m)"", "b\na\n", "y", "R"($-[0])"", "1");
        test(regex_16, "16", "R"('\Z'm)"", "b\na", "y", "R"($-[0])"", "3");
        test(regex_17, "17", "R"('\z'm)"", "b\na", "y", "R"($-[0])"", "3");
        test(regex_18, "18", "R"('$'m)"", "b\na", "y", "R"($-[0])"", "1");
        test(regex_19, "19", "R"(a\Z)"", "a\nb\n", "n", "R"(-)"", "-");
        test(regex_20, "20", "R"(a\z)"", "a\nb\n", "n", "R"(-)"", "-");
        test(regex_21, "21", "R"(a$)"", "a\nb\n", "n", "R"(-)"", "-");
        test(regex_22, "22", "R"(a\Z)"", "b\na\n", "y", "R"($-[0])"", "2");
        test(regex_23, "23", "R"(a\z)"", "b\na\n", "n", "R"(-)"", "-");
        test(regex_24, "24", "R"(a$)"", "b\na\n", "y", "R"($-[0])"", "2");
        test(regex_25, "25", "R"(a\Z)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_26, "26", "R"(a\z)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_27, "27", "R"(a$)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_28, "28", "R"('a\Z'm)"", "a\nb\n", "n", "R"(-)"", "-");
        test(regex_29, "29", "R"('a\z'm)"", "a\nb\n", "n", "R"(-)"", "-");
        test(regex_30, "30", "R"('a$'m)"", "a\nb\n", "y", "R"($-[0])"", "0");
        test(regex_31, "31", "R"('a\Z'm)"", "b\na\n", "y", "R"($-[0])"", "2");
        test(regex_32, "32", "R"('a\z'm)"", "b\na\n", "n", "R"(-)"", "-");
        test(regex_33, "33", "R"('a$'m)"", "b\na\n", "y", "R"($-[0])"", "2");
        test(regex_34, "34", "R"('a\Z'm)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_35, "35", "R"('a\z'm)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_36, "36", "R"('a$'m)"", "b\na", "y", "R"($-[0])"", "2");
        test(regex_37, "37", "R"(aa\Z)"", "aa\nb\n", "n", "R"(-)"", "-");
        test(regex_38, "38", "R"(aa\z)"", "aa\nb\n", "n", "R"(-)"", "-");
        test(regex_39, "39", "R"(aa$)"", "aa\nb\n", "n", "R"(-)"", "-");
        test(regex_40, "40", "R"(aa\Z)"", "b\naa\n", "y", "R"($-[0])"", "2");
        test(regex_41, "41", "R"(aa\z)"", "b\naa\n", "n", "R"(-)"", "-");
        test(regex_42, "42", "R"(aa$)"", "b\naa\n", "y", "R"($-[0])"", "2");
        test(regex_43, "43", "R"(aa\Z)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_44, "44", "R"(aa\z)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_45, "45", "R"(aa$)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_46, "46", "R"('aa\Z'm)"", "aa\nb\n", "n", "R"(-)"", "-");
        test(regex_47, "47", "R"('aa\z'm)"", "aa\nb\n", "n", "R"(-)"", "-");
        test(regex_48, "48", "R"('aa$'m)"", "aa\nb\n", "y", "R"($-[0])"", "0");
        test(regex_49, "49", "R"('aa\Z'm)"", "b\naa\n", "y", "R"($-[0])"", "2");
        test(regex_50, "50", "R"('aa\z'm)"", "b\naa\n", "n", "R"(-)"", "-");
        test(regex_51, "51", "R"('aa$'m)"", "b\naa\n", "y", "R"($-[0])"", "2");
        test(regex_52, "52", "R"('aa\Z'm)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_53, "53", "R"('aa\z'm)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_54, "54", "R"('aa$'m)"", "b\naa", "y", "R"($-[0])"", "2");
        test(regex_55, "55", "R"(aa\Z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_56, "56", "R"(aa\z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_57, "57", "R"(aa$)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_58, "58", "R"(aa\Z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_59, "59", "R"(aa\z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_60, "60", "R"(aa$)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_61, "61", "R"(aa\Z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_62, "62", "R"(aa\z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_63, "63", "R"(aa$)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_64, "64", "R"('aa\Z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_65, "65", "R"('aa\z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_66, "66", "R"('aa$'m)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_67, "67", "R"('aa\Z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_68, "68", "R"('aa\z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_69, "69", "R"('aa$'m)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_70, "70", "R"('aa\Z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_71, "71", "R"('aa\z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_72, "72", "R"('aa$'m)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_73, "73", "R"(aa\Z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_74, "74", "R"(aa\z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_75, "75", "R"(aa$)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_76, "76", "R"(aa\Z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_77, "77", "R"(aa\z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_78, "78", "R"(aa$)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_79, "79", "R"(aa\Z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_80, "80", "R"(aa\z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_81, "81", "R"(aa$)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_82, "82", "R"('aa\Z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_83, "83", "R"('aa\z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_84, "84", "R"('aa$'m)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_85, "85", "R"('aa\Z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_86, "86", "R"('aa\z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_87, "87", "R"('aa$'m)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_88, "88", "R"('aa\Z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_89, "89", "R"('aa\z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_90, "90", "R"('aa$'m)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_91, "91", "R"(ab\Z)"", "ab\nb\n", "n", "R"(-)"", "-");
        test(regex_92, "92", "R"(ab\z)"", "ab\nb\n", "n", "R"(-)"", "-");
        test(regex_93, "93", "R"(ab$)"", "ab\nb\n", "n", "R"(-)"", "-");
        test(regex_94, "94", "R"(ab\Z)"", "b\nab\n", "y", "R"($-[0])"", "2");
        test(regex_95, "95", "R"(ab\z)"", "b\nab\n", "n", "R"(-)"", "-");
        test(regex_96, "96", "R"(ab$)"", "b\nab\n", "y", "R"($-[0])"", "2");
        test(regex_97, "97", "R"(ab\Z)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_98, "98", "R"(ab\z)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_99, "99", "R"(ab$)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_100, "100", "R"('ab\Z'm)"", "ab\nb\n", "n", "R"(-)"", "-");
        test(regex_101, "101", "R"('ab\z'm)"", "ab\nb\n", "n", "R"(-)"", "-");
        test(regex_102, "102", "R"('ab$'m)"", "ab\nb\n", "y", "R"($-[0])"", "0");
        test(regex_103, "103", "R"('ab\Z'm)"", "b\nab\n", "y", "R"($-[0])"", "2");
        test(regex_104, "104", "R"('ab\z'm)"", "b\nab\n", "n", "R"(-)"", "-");
        test(regex_105, "105", "R"('ab$'m)"", "b\nab\n", "y", "R"($-[0])"", "2");
        test(regex_106, "106", "R"('ab\Z'm)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_107, "107", "R"('ab\z'm)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_108, "108", "R"('ab$'m)"", "b\nab", "y", "R"($-[0])"", "2");
        test(regex_109, "109", "R"(ab\Z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_110, "110", "R"(ab\z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_111, "111", "R"(ab$)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_112, "112", "R"(ab\Z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_113, "113", "R"(ab\z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_114, "114", "R"(ab$)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_115, "115", "R"(ab\Z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_116, "116", "R"(ab\z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_117, "117", "R"(ab$)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_118, "118", "R"('ab\Z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_119, "119", "R"('ab\z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_120, "120", "R"('ab$'m)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_121, "121", "R"('ab\Z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_122, "122", "R"('ab\z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_123, "123", "R"('ab$'m)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_124, "124", "R"('ab\Z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_125, "125", "R"('ab\z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_126, "126", "R"('ab$'m)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_127, "127", "R"(ab\Z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_128, "128", "R"(ab\z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_129, "129", "R"(ab$)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_130, "130", "R"(ab\Z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_131, "131", "R"(ab\z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_132, "132", "R"(ab$)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_133, "133", "R"(ab\Z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_134, "134", "R"(ab\z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_135, "135", "R"(ab$)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_136, "136", "R"('ab\Z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_137, "137", "R"('ab\z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_138, "138", "R"('ab$'m)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_139, "139", "R"('ab\Z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_140, "140", "R"('ab\z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_141, "141", "R"('ab$'m)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_142, "142", "R"('ab\Z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_143, "143", "R"('ab\z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_144, "144", "R"('ab$'m)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_145, "145", "R"(abb\Z)"", "abb\nb\n", "n", "R"(-)"", "-");
        test(regex_146, "146", "R"(abb\z)"", "abb\nb\n", "n", "R"(-)"", "-");
        test(regex_147, "147", "R"(abb$)"", "abb\nb\n", "n", "R"(-)"", "-");
        test(regex_148, "148", "R"(abb\Z)"", "b\nabb\n", "y", "R"($-[0])"", "2");
        test(regex_149, "149", "R"(abb\z)"", "b\nabb\n", "n", "R"(-)"", "-");
        test(regex_150, "150", "R"(abb$)"", "b\nabb\n", "y", "R"($-[0])"", "2");
        test(regex_151, "151", "R"(abb\Z)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_152, "152", "R"(abb\z)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_153, "153", "R"(abb$)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_154, "154", "R"('abb\Z'm)"", "abb\nb\n", "n", "R"(-)"", "-");
        test(regex_155, "155", "R"('abb\z'm)"", "abb\nb\n", "n", "R"(-)"", "-");
        test(regex_156, "156", "R"('abb$'m)"", "abb\nb\n", "y", "R"($-[0])"", "0");
        test(regex_157, "157", "R"('abb\Z'm)"", "b\nabb\n", "y", "R"($-[0])"", "2");
        test(regex_158, "158", "R"('abb\z'm)"", "b\nabb\n", "n", "R"(-)"", "-");
        test(regex_159, "159", "R"('abb$'m)"", "b\nabb\n", "y", "R"($-[0])"", "2");
        test(regex_160, "160", "R"('abb\Z'm)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_161, "161", "R"('abb\z'm)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_162, "162", "R"('abb$'m)"", "b\nabb", "y", "R"($-[0])"", "2");
        test(regex_163, "163", "R"(abb\Z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_164, "164", "R"(abb\z)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_165, "165", "R"(abb$)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_166, "166", "R"(abb\Z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_167, "167", "R"(abb\z)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_168, "168", "R"(abb$)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_169, "169", "R"(abb\Z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_170, "170", "R"(abb\z)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_171, "171", "R"(abb$)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_172, "172", "R"('abb\Z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_173, "173", "R"('abb\z'm)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_174, "174", "R"('abb$'m)"", "ac\nb\n", "n", "R"(-)"", "-");
        test(regex_175, "175", "R"('abb\Z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_176, "176", "R"('abb\z'm)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_177, "177", "R"('abb$'m)"", "b\nac\n", "n", "R"(-)"", "-");
        test(regex_178, "178", "R"('abb\Z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_179, "179", "R"('abb\z'm)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_180, "180", "R"('abb$'m)"", "b\nac", "n", "R"(-)"", "-");
        test(regex_181, "181", "R"(abb\Z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_182, "182", "R"(abb\z)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_183, "183", "R"(abb$)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_184, "184", "R"(abb\Z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_185, "185", "R"(abb\z)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_186, "186", "R"(abb$)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_187, "187", "R"(abb\Z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_188, "188", "R"(abb\z)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_189, "189", "R"(abb$)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_190, "190", "R"('abb\Z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_191, "191", "R"('abb\z'm)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_192, "192", "R"('abb$'m)"", "ca\nb\n", "n", "R"(-)"", "-");
        test(regex_193, "193", "R"('abb\Z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_194, "194", "R"('abb\z'm)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_195, "195", "R"('abb$'m)"", "b\nca\n", "n", "R"(-)"", "-");
        test(regex_196, "196", "R"('abb\Z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_197, "197", "R"('abb\z'm)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_198, "198", "R"('abb$'m)"", "b\nca", "n", "R"(-)"", "-");
        test(regex_199, "199", "R"('\Aa$'m)"", "a\n\n", "y", "R"($&)"", "a");
        std::cout << std::endl;
    }
    
    // @regex metafunction: compile-time regex validation
    // Note: regex members are compiled at construction
};

auto main() -> void {
    test_tests_14_multiline_modifier().run();
}

