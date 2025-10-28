#include "tblgen_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>

// Simple JSON parser for tblgen output (no external deps)
namespace {

std::string extract_string_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.length();
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";

    return json.substr(pos, end - pos);
}

std::vector<std::string> extract_array(const std::string& json, const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\":[";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos += search.length();
    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string array_content = json.substr(pos, end - pos);

    // Parse array elements
    size_t elem_pos = 0;
    while (elem_pos < array_content.length()) {
        size_t quote_start = array_content.find("\"", elem_pos);
        if (quote_start == std::string::npos) break;

        size_t quote_end = array_content.find("\"", quote_start + 1);
        if (quote_end == std::string::npos) break;

        result.push_back(array_content.substr(quote_start + 1, quote_end - quote_start - 1));
        elem_pos = quote_end + 1;
    }

    return result;
}

std::string extract_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":{";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.length() - 1; // Include the opening brace
    int brace_count = 0;
    size_t end = pos;

    while (end < json.length()) {
        if (json[end] == '{') brace_count++;
        else if (json[end] == '}') {
            brace_count--;
            if (brace_count == 0) break;
        }
        end++;
    }

    if (brace_count != 0) return "";
    return json.substr(pos, end - pos + 1);
}

} // namespace

namespace cppfort::stage0 {

bool TblgenLoader::load_json(const std::string& path) {
    units_.clear();

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open tblgen JSON: " << path << "\n";
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // Parse each semantic unit
    std::vector<std::string> unit_names = {"FunctionDecl", "VariableDecl", "TypeAlias", "Parameter"};

    for (const auto& unit_name : unit_names) {
        std::string unit_json = extract_object(json, unit_name);
        if (unit_json.empty()) continue;

        TblgenSemanticUnit unit;
        unit.name = extract_string_value(unit_json, "Name");
        unit.segments = extract_array(unit_json, "Segments");
        unit.c_pattern = extract_string_value(unit_json, "C_pattern");
        unit.cpp_pattern = extract_string_value(unit_json, "CPP_pattern");
        unit.cpp2_pattern = extract_string_value(unit_json, "CPP2_pattern");

        if (!unit.name.empty()) {
            units_[unit_name] = unit;
        }
    }

    return !units_.empty();
}

const TblgenSemanticUnit* TblgenLoader::get_unit(const std::string& name) const {
    auto it = units_.find(name);
    return (it != units_.end()) ? &it->second : nullptr;
}

} // namespace cppfort::stage0
