#pragma once

#include <string>
#include <unordered_set>
#include <optional>

namespace cppfort::stage0 {

enum class CModeRouting {
    Passthrough,
    Inherit,
    Specialize,
    Preserve
};

class CModeRouter {
public:
    static std::optional<CModeRouting> route(std::string_view patternName) {
        if (isPassthrough(patternName)) return CModeRouting::Passthrough;
        if (isInherit(patternName)) return CModeRouting::Inherit;
        if (isSpecialize(patternName)) return CModeRouting::Specialize;
        if (isPreserve(patternName)) return CModeRouting::Preserve;
        return std::nullopt;
    }

private:
    static bool isPassthrough(std::string_view name) {
        static const std::unordered_set<std::string> patterns = {
            "cpp1_parameter_normal", "cpp1_function_decl", "cpp1_struct_def",
            "cpp1_enum_def", "cpp1_typedef", "c_simple_types"
        };
        return patterns.find(std::string(name)) != patterns.end();
    }

    static bool isInherit(std::string_view name) {
        static const std::unordered_set<std::string> patterns = {
            "c_struct_with_initializers", "c_static_inline",
            "c_complex_numbers", "c_designated_initializers"
        };
        return patterns.find(std::string(name)) != patterns.end();
    }

    static bool isSpecialize(std::string_view name) {
        static const std::unordered_set<std::string> patterns = {
            "c_kandr_prototypes", "c_void_star_everywhere",
            "c_setjmp_longjmp", "c_manual_vtables"
        };
        return patterns.find(std::string(name)) != patterns.end();
    }

    static bool isPreserve(std::string_view name) {
        static const std::unordered_set<std::string> patterns = {
            "c_restrict_pointers", "c_variable_length_arrays",
            "c_flexible_array_members"
        };
        return patterns.find(std::string(name)) != patterns.end();
    }
};

} // namespace cppfort::stage0
