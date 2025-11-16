#include "json_pattern_loader.h"
#include <fstream>
#include <algorithm>
#include <cctype>

namespace cppfort::stage0 {

// Simple JSON parsing implementation - sufficient for pattern files

std::string_view JsonPatternLoader::trim(std::string_view str) const {
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) start++;
    
    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) end--;
    
    return str.substr(start, end - start);
}

size_t JsonPatternLoader::skip_whitespace(const std::string& json, size_t pos) const {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        pos++;
    }
    return pos;
}

size_t JsonPatternLoader::find_matching_brace(const std::string& json, size_t start) const {
    char open = json[start];
    char close = (open == '{') ? '}' : (open == '[') ? ']' : (open == '"') ? '"' : '\0';
    
    if (close == '\0') return start;
    
    int depth = 1;
    bool in_string = (open == '"');
    
    for (size_t i = start + 1; i < json.size(); i++) {
        char ch = json[i];
        
        if (in_string) {
            if (ch == '"' && json[i-1] != '\\') {
                return i;
            }
        } else {
            if (open != '"' && ch == '"') {
                in_string = true;
            } else if (ch == open) {
                depth++;
            } else if (ch == close) {
                depth--;
                if (depth == 0) return i;
            }
        }
    }
    return json.size();
}

std::string JsonPatternLoader::extract_string(const std::string& json, size_t& pos) {
    pos = skip_whitespace(json, pos);
    
    if (pos >= json.size() || json[pos] != '"') {
        last_error = "Expected string starting with \"";
        return "";
    }
    
    size_t end = find_matching_brace(json, pos);
    if (end >= json.size()) {
        last_error = "Unterminated string";
        return "";
    }
    
    std::string result = json.substr(pos + 1, end - pos - 1);
    pos = end + 1;
    
    // Unescape simple escape sequences
    size_t escape = result.find("\\\"");
    while (escape != std::string::npos) {
        result.replace(escape, 2, "\"");
        escape = result.find("\\\"", escape + 1);
    }
    
    return result;
}

int JsonPatternLoader::extract_int(const std::string& json, size_t& pos) {
    pos = skip_whitespace(json, pos);
    
    size_t start = pos;
    while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) || 
           json[pos] == '-')) {
        pos++;
    }
    
    if (pos == start) {
        last_error = "Expected integer";
        return 0;
    }
    
    return std::stoi(json.substr(start, pos - start));
}

std::vector<std::string> JsonPatternLoader::extract_string_array(const std::string& json, size_t& pos) {
    std::vector<std::string> result;
    
    pos = skip_whitespace(json, pos);
    if (pos >= json.size() || json[pos] != '[') {
        last_error = "Expected array starting with [";
        return result;
    }
    
    pos++; // Skip [
    
    while (pos < json.size()) {
        pos = skip_whitespace(json, pos);
        if (pos >= json.size()) break;
        
        if (json[pos] == ']') {
            pos++;
            return result;
        }
        
        if (json[pos] == ',') {
            pos++;
            continue;
        }
        
        std::string item = extract_string(json, pos);
        if (!item.empty() || last_error.empty()) {
            result.push_back(item);
        }
    }
    
    last_error = "Unterminated array";
    return result;
}

std::unordered_map<int, std::string> JsonPatternLoader::extract_templates_object(const std::string& json, size_t& pos) {
    std::unordered_map<int, std::string> result;
    
    pos = skip_whitespace(json, pos);
    if (pos >= json.size() || json[pos] != '{') {
        last_error = "Expected object starting with {";
        return result;
    }
    
    pos++; // Skip {
    
    while (pos < json.size()) {
        pos = skip_whitespace(json, pos);
        if (pos >= json.size()) break;
        
        if (json[pos] == '}') {
            pos++;
            return result;
        }
        
        if (json[pos] == ',') {
            pos++;
            continue;
        }
        
        // Key is an integer in quotes
        std::string key_str = extract_string(json, pos);
        if (!key_str.empty()) {
            int key = std::stoi(key_str);
            
            pos = skip_whitespace(json, pos);
            if (pos < json.size() && json[pos] == ':') pos++;
            
            std::string value = extract_string(json, pos);
            result[key] = value;
        }
    }
    
    last_error = "Unterminated object";
    return result;
}

std::vector<JsonParsedPattern> JsonPatternLoader::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        last_error = "Failed to open file: " + filepath;
        return {};
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    return load_from_string(content);
}

std::vector<JsonParsedPattern> JsonPatternLoader::load_from_string(const std::string& json) {
    std::vector<JsonParsedPattern> patterns;
    last_error.clear();
    
    size_t pos = 0;
    pos = skip_whitespace(json, pos);
    
    if (pos >= json.size() || json[pos] != '{') {
        last_error = "Expected JSON object starting with {";
        return {};
    }
    
    pos++; // Skip {
    
    // Find "patterns" key
    while (pos < json.size()) {
        pos = skip_whitespace(json, pos);
        
        if (pos < json.size() && json[pos] == '}') {
            pos++;
            break;
        }
        
        if (pos < json.size() && json[pos] == '"') {
            std::string key = extract_string(json, pos);
            
            pos = skip_whitespace(json, pos);
            if (pos < json.size() && json[pos] == ':') pos++;
            
            if (key == "patterns") {
                // Parse patterns array
                pos = skip_whitespace(json, pos);
                if (pos >= json.size() || json[pos] != '[') {
                    last_error = "Expected patterns array starting with [";
                    return {};
                }
                pos++; // Skip [
                
                while (pos < json.size()) {
                    pos = skip_whitespace(json, pos);
                    if (pos >= json.size()) break;
                    
                    if (json[pos] == ']') {
                        pos++;
                        break;
                    }
                    
                    if (json[pos] == ',') {
                        pos++;
                        continue;
                    }
                    
                    if (json[pos] == '{') {
                        // Parse pattern object
                        JsonParsedPattern pattern;
                        pos++; // Skip {
                        
                        while (pos < json.size()) {
                            pos = skip_whitespace(json, pos);
                            if (pos >= json.size()) break;
                            
                            if (json[pos] == '}') {
                                pos++;
                                patterns.push_back(pattern);
                                break;
                            }
                            
                            if (json[pos] == ',') {
                                pos++;
                                continue;
                            }
                            
                            if (json[pos] == '"') {
                                std::string field = extract_string(json, pos);
                                
                                pos = skip_whitespace(json, pos);
                                if (pos < json.size() && json[pos] == ':') pos++;
                                
                                if (field == "name") {
                                    pattern.name = extract_string(json, pos);
                                } else if (field == "use_alternating") {
                                    pos = skip_whitespace(json, pos);
                                    if (pos + 4 < json.size() && json.substr(pos, 4) == "true") {
                                        pattern.use_alternating = true;
                                        pos += 4;
                                    } else if (pos + 5 < json.size() && json.substr(pos, 5) == "false") {
                                        pattern.use_alternating = false;
                                        pos += 5;
                                    }
                                } else if (field == "alternating_anchors") {
                                    pattern.alternating_anchors = extract_string_array(json, pos);
                                } else if (field == "evidence_types") {
                                    pattern.evidence_types = extract_string_array(json, pos);
                                } else if (field == "grammar_modes") {
                                    pattern.grammar_modes = extract_int(json, pos);
                                } else if (field == "priority") {
                                    pattern.priority = extract_int(json, pos);
                                } else if (field == "transformation_templates") {
                                    pattern.templates = extract_templates_object(json, pos);
                                } else {
                                    // Skip unknown field
                                    pos = skip_whitespace(json, pos);
                                    if (pos < json.size() && json[pos] == '"') {
                                        extract_string(json, pos);
                                    } else if (pos < json.size() && (json[pos] == '{' || json[pos] == '[')) {
                                        pos = find_matching_brace(json, pos) + 1;
                                    } else {
                                        while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }
    
    return patterns;
}

} // namespace cppfort::stage0