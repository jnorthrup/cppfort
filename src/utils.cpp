#include "utils.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <random>

namespace cpp2_transpiler {

int Utils::unique_counter = 0;

// String utilities
std::string Utils::escape_string(const std::string& str) {
    std::string result;
    result.reserve(str.length() * 2);

    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c >= 32 && c <= 126) {
                    result += c;
                } else {
                    result += std::format("\\x{:02x}", static_cast<unsigned char>(c));
                }
                break;
        }
    }

    return result;
}

std::string Utils::unescape_string(const std::string& str) {
    std::string result;
    result.reserve(str.length());

    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == '\\' && i + 1 < str.length()) {
            switch (str[i + 1]) {
                case '"': result += '"'; ++i; break;
                case '\\': result += '\\'; ++i; break;
                case 'n': result += '\n'; ++i; break;
                case 'r': result += '\r'; ++i; break;
                case 't': result += '\t'; ++i; break;
                case 'x':
                    if (i + 3 < str.length()) {
                        std::string hex = str.substr(i + 2, 2);
                        result += static_cast<char>(std::stoi(hex, nullptr, 16));
                        i += 3;
                    }
                    break;
                default:
                    result += str[i];
                    break;
            }
        } else {
            result += str[i];
        }
    }

    return result;
}

std::string Utils::join_strings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) return "";

    std::ostringstream result;
    result << strings[0];

    for (size_t i = 1; i < strings.size(); ++i) {
        result << delimiter << strings[i];
    }

    return result.str();
}

std::vector<std::string> Utils::split_string(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::istringstream stream(str);
    std::string token;

    while (std::getline(stream, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

std::string Utils::trim_string(const std::string& str) {
    auto start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";

    auto end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::string Utils::to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

std::string Utils::to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](char c) { return static_cast<char>(std::toupper(c)); });
    return result;
}

bool Utils::starts_with(const std::string& str, const std::string& prefix) {
    return str.length() >= prefix.length() &&
           str.substr(0, prefix.length()) == prefix;
}

bool Utils::ends_with(const std::string& str, const std::string& suffix) {
    return str.length() >= suffix.length() &&
           str.substr(str.length() - suffix.length()) == suffix;
}

std::string Utils::replace_all(const std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;

    std::string result = str;
    size_t pos = 0;

    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }

    return result;
}

// File utilities
std::string Utils::read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

bool Utils::write_file(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    if (!file) return false;
    file << content;
    return file.good();
}

bool Utils::file_exists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

std::string Utils::get_file_extension(const std::string& filename) {
    auto pos = filename.find_last_of('.');
    return pos != std::string::npos ? filename.substr(pos) : "";
}

std::string Utils::get_file_basename(const std::string& filename) {
    auto pos = filename.find_last_of('.');
    return pos != std::string::npos ? filename.substr(0, pos) : filename;
}

// Name utilities
std::string Utils::mangle_name(const std::string& name) {
    // Simple name mangling for unique identifiers
    return "_" + name + "_mangled";
}

std::string Utils::demangle_name(const std::string& mangled_name) {
    // Simple demangling
    if (starts_with(mangled_name, "_") && ends_with(mangled_name, "_mangled")) {
        return mangled_name.substr(1, mangled_name.length() - 9);
    }
    return mangled_name;
}

std::string Utils::generate_unique_id(const std::string& prefix) {
    return prefix + std::to_string(++unique_counter);
}

bool Utils::is_valid_identifier(const std::string& str) {
    if (str.empty()) return false;

    if (!std::isalpha(str[0]) && str[0] != '_') return false;

    for (char c : str) {
        if (!std::isalnum(c) && c != '_') return false;
    }

    return true;
}

std::string Utils::make_valid_identifier(const std::string& str) {
    std::string result;
    result.reserve(str.length());

    // First character
    if (!str.empty() && (std::isalpha(str[0]) || str[0] == '_')) {
        result += str[0];
    } else if (!str.empty()) {
        result += '_';
        result += str[0];
    }

    // Remaining characters
    for (size_t i = 1; i < str.length(); ++i) {
        if (std::isalnum(str[i]) || str[i] == '_') {
            result += str[i];
        } else {
            result += '_';
        }
    }

    return result;
}

// Type utilities
bool Utils::is_builtin_type(const std::string& type_name) {
    static const std::vector<std::string> builtins = {
        "bool", "char", "signed char", "unsigned char",
        "short", "unsigned short", "int", "unsigned int",
        "long", "unsigned long", "long long", "unsigned long long",
        "float", "double", "long double", "void", "auto",
        "size_t", "ptrdiff_t", "nullptr_t"
    };

    return std::find(builtins.begin(), builtins.end(), type_name) != builtins.end();
}

bool Utils::is_pointer_type(const std::string& type_name) {
    return ends_with(type_name, "*") || ends_with(type_name, "* const");
}

bool Utils::is_reference_type(const std::string& type_name) {
    return ends_with(type_name, "&") || ends_with(type_name, "&&");
}

bool Utils::is_const_type(const std::string& type_name) {
    return starts_with(type_name, "const ") || ends_with(type_name, " const");
}

std::string Utils::remove_cv_qualifiers(const std::string& type_name) {
    std::string result = type_name;

    // Remove leading const
    if (starts_with(result, "const ")) {
        result = result.substr(6);
    }

    // Remove trailing const
    if (ends_with(result, " const")) {
        result = result.substr(0, result.length() - 6);
    }

    return trim_string(result);
}

std::string Utils::get_base_type(const std::string& type_name) {
    std::string base = remove_cv_qualifiers(type_name);

    // Remove pointer and reference qualifiers
    while (ends_with(base, "*") || ends_with(base, "&")) {
        base = base.substr(0, base.length() - 1);
        base = trim_string(base);
    }

    return base;
}

// C++ specific utilities
std::string Utils::get_cpp_keyword_map(const std::string& cpp2_keyword) {
    static const std::unordered_map<std::string, std::string> keyword_map = {
        {"func", "function"},
        {"let", "const"},
        {"mut", ""},
        {"type", "struct"},
        {"namespace", "namespace"},
        {"return", "return"},
        {"if", "if"},
        {"else", "else"},
        {"while", "while"},
        {"for", "for"},
        {"break", "break"},
        {"continue", "continue"},
        {"switch", "switch"},
        {"case", "case"},
        {"default", "default"},
        {"try", "try"},
        {"catch", "catch"},
        {"throw", "throw"},
        {"import", "#include"},
        {"export", "export"},
        {"public", "public"},
        {"private", "private"},
        {"protected", "protected"},
        {"virtual", "virtual"},
        {"override", "override"},
        {"final", "final"},
        {"explicit", "explicit"},
        {"static", "static"},
        {"inline", "inline"},
        {"constexpr", "constexpr"},
        {"consteval", "consteval"},
        {"noexcept", "noexcept"}
    };

    auto it = keyword_map.find(cpp2_keyword);
    return it != keyword_map.end() ? it->second : cpp2_keyword;
}

std::vector<std::string> Utils::get_required_headers(const std::string& cpp_feature) {
    static const std::unordered_map<std::string, std::vector<std::string>> feature_headers = {
        {"format", {"<format>"}},
        {"print", {"<iostream>"}},
        {"vector", {"<vector>"}},
        {"string", {"<string>"}},
        {"span", {"<span>"}},
        {"ranges", {"<ranges>"}},
        {"algorithm", {"<algorithm>"}},
        {"memory", {"<memory>"}},
        {"optional", {"<optional>"}},
        {"variant", {"<variant>"}},
        {"tuple", {"<tuple>"}},
        {"chrono", {"<chrono>"}},
        {"filesystem", {"<filesystem>"}},
        {"thread", {"<thread>"}},
        {"mutex", {"<mutex>"}},
        {"atomic", {"<atomic>"}},
        {"contract", {"<contract>"}}
    };

    auto it = feature_headers.find(cpp_feature);
    return it != feature_headers.end() ? it->second : std::vector<std::string>{};
}

std::string Utils::format_cpp_declaration(const std::string& name, const std::string& type) {
    if (type.empty()) return name + ";";
    return type + " " + name + ";";
}

std::string Utils::generate_include_guard(const std::string& filename) {
    std::string guard = "INCLUDE_" + to_upper(get_file_basename(filename));
    guard = replace_all(guard, ".", "_");
    guard = replace_all(guard, "/", "_");
    guard = replace_all(guard, "\\", "_");
    return guard;
}

} // namespace cpp2_transpiler