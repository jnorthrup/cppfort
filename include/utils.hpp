#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <format>

namespace cpp2_transpiler {

class Utils {
public:
    // String utilities
    static std::string escape_string(const std::string& str);
    static std::string unescape_string(const std::string& str);
    static std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter);
    static std::vector<std::string> split_string(const std::string& str, char delimiter);
    static std::string trim_string(const std::string& str);
    static std::string to_lower(const std::string& str);
    static std::string to_upper(const std::string& str);
    static bool starts_with(const std::string& str, const std::string& prefix);
    static bool ends_with(const std::string& str, const std::string& suffix);
    static std::string replace_all(const std::string& str, const std::string& from, const std::string& to);

    // File utilities
    static std::string read_file(const std::string& filename);
    static bool write_file(const std::string& filename, const std::string& content);
    static bool file_exists(const std::string& filename);
    static std::string get_file_extension(const std::string& filename);
    static std::string get_file_basename(const std::string& filename);

    // Name utilities
    static std::string mangle_name(const std::string& name);
    static std::string demangle_name(const std::string& mangled_name);
    static std::string generate_unique_id(const std::string& prefix = "");
    static bool is_valid_identifier(const std::string& str);
    static std::string make_valid_identifier(const std::string& str);

    // Type utilities
    static bool is_builtin_type(const std::string& type_name);
    static bool is_pointer_type(const std::string& type_name);
    static bool is_reference_type(const std::string& type_name);
    static bool is_const_type(const std::string& type_name);
    static std::string remove_cv_qualifiers(const std::string& type_name);
    static std::string get_base_type(const std::string& type_name);

    // C++ specific utilities
    static std::string get_cpp_keyword_map(const std::string& cpp2_keyword);
    static std::vector<std::string> get_required_headers(const std::string& cpp_feature);
    static std::string format_cpp_declaration(const std::string& name, const std::string& type);
    static std::string generate_include_guard(const std::string& filename);

private:
    static int unique_counter;
};

} // namespace cpp2_transpiler