#include "tblgen_patterns.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace cppfort::ir {
namespace fs = ::std::filesystem;

bool PatternDatabase::loadFromYaml(const ::std::filesystem::path& filePath) {
    return loadYamlFile(filePath.string());
}

bool PatternDatabase::loadYamlFile(const ::std::string& filepath) {
    ::std::ifstream file(filepath);
    if (!file.is_open()) {
        ::std::cerr << "Failed to open file: " << filepath << ::std::endl;
        return false;
    }

    ::std::string content((::std::istreambuf_iterator<char>(file)),
                         ::std::istreambuf_iterator<char>());
    file.close();

    // Parse all patterns from the file
    ::std::istringstream iss(content);
    ::std::string line;
    ::std::vector<::std::string> patternChunks;
    ::std::string currentChunk;
    bool inPatternList = false;

    while (::std::getline(iss, line)) {
        // Skip empty lines and comments
        ::std::string trimmed = line;
        while (!trimmed.empty() && ::std::isspace(trimmed.front())) {
            trimmed = trimmed.substr(1);
        }

        if (trimmed.find("cpp2_canonical_patterns:") == 0) {
            inPatternList = true;
            continue;
        }

        if (!inPatternList) continue;

        // Detect start of new pattern (line starting with "  -")
        if (line.size() >= 3 && line[0] == ' ' && line[1] == ' ' && line[2] == '-') {
            if (!currentChunk.empty()) {
                patternChunks.push_back(currentChunk);
            }
            currentChunk = line + "\n";
        } else if (!currentChunk.empty()) {
            currentChunk += line + "\n";
        }
    }

    // Add last chunk
    if (!currentChunk.empty()) {
        patternChunks.push_back(currentChunk);
    }

    // Parse each pattern chunk
    int parsedCount = 0;
    for (const auto& chunk : patternChunks) {
        auto pattern = parsePattern(chunk);
        if (pattern) {
            _patterns[pattern->name] = *pattern;
            _patternsByOrbit[pattern->orbit_id].push_back(*pattern);
            parsedCount++;
        }
    }

    if (parsedCount > 0) {
        ::std::cerr << "Loaded " << parsedCount << " patterns from " << filepath << ::std::endl;
        return true;
    }

    ::std::cerr << "Failed to parse any patterns from file: " << filepath << ::std::endl;
    return false;
}

::std::optional<OrbitPattern> PatternDatabase::parsePattern(const ::std::string& yamlContent) {
    // Parser for cpp2_canonical_patterns format:
    // cpp2_canonical_patterns:
    //   - name: pattern_name
    //     canonical_form: "x: int = 5"
    //     pattern_type: type_annotation
    //     confidence: 1.0
    //     scope_filter: [list]

    OrbitPattern pattern;
    ::std::istringstream iss(yamlContent);
    ::std::string line;
    ::std::string currentSection;
    bool inPattern = false;
    uint32_t autoOrbitId = 1;

    while (::std::getline(iss, line)) {
        // Remove trailing whitespace
        while (!line.empty() && ::std::isspace(line.back())) {
            line.pop_back();
        }

        if (line.empty() || line[0] == '#') continue;

        // Detect pattern start
        size_t indent = 0;
        while (indent < line.size() && ::std::isspace(line[indent])) {
            ++indent;
        }
        ::std::string trimmed = line.substr(indent);

        // Start of new pattern item
        if (trimmed[0] == '-' && trimmed.find("name:") != ::std::string::npos) {
            if (inPattern && !pattern.name.empty()) {
                return pattern; // Return first pattern found
            }
            inPattern = true;
            currentSection.clear();

            // Extract name from same line if present
            size_t namePos = trimmed.find("name:");
            if (namePos != ::std::string::npos) {
                ::std::string value = trimmed.substr(namePos + 5);
                size_t valueStart = value.find_first_not_of(" \t");
                if (valueStart != ::std::string::npos) {
                    value = value.substr(valueStart);
                    if (!value.empty() && value.front() == '"' && value.back() == '"') {
                        value = value.substr(1, value.size() - 2);
                    }
                    pattern.name = value;
                    // Generate orbit_id from name hash
                    ::std::hash<::std::string> hasher;
                    pattern.orbit_id = static_cast<uint32_t>(hasher(value) % 10000);
                }
            }
            continue;
        }

        if (!inPattern) continue;

        if (trimmed.find("name:") == 0 && trimmed[0] != '-') {
            ::std::string value = trimmed.substr(5);
            size_t valueStart = value.find_first_not_of(" \t");
            if (valueStart != ::std::string::npos) {
                value = value.substr(valueStart);
                if (!value.empty() && value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.size() - 2);
                }
                pattern.name = value;
                ::std::hash<::std::string> hasher;
                pattern.orbit_id = static_cast<uint32_t>(hasher(value) % 10000);
            }
        } else if (trimmed.find("canonical_form:") == 0) {
            ::std::string value = trimmed.substr(15);
            size_t valueStart = value.find_first_not_of(" \t");
            if (valueStart != ::std::string::npos) {
                value = value.substr(valueStart);
                if (!value.empty() && value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.size() - 2);
                }
                pattern.signature_patterns.push_back(value);
            }
        } else if (trimmed.find("confidence:") == 0) {
            ::std::string value = trimmed.substr(11);
            size_t valueStart = value.find_first_not_of(" \t");
            if (valueStart != ::std::string::npos) {
                value = value.substr(valueStart);
                try {
                    pattern.weight = ::std::stod(value);
                } catch (...) {
                    pattern.weight = 1.0;
                }
            }
        } else if (trimmed.find("scope_filter:") == 0) {
            currentSection = "scope";
        } else if (trimmed[0] == '-' && currentSection == "scope") {
            ::std::string value = trimmed.substr(1);
            size_t valueStart = value.find_first_not_of(" \t");
            if (valueStart != ::std::string::npos) {
                value = value.substr(valueStart);
                pattern.protocol_indicators.push_back(value);
            }
        }
    }

    if (!pattern.name.empty()) {
        return pattern;
    }

    return ::std::nullopt;
}

bool PatternDatabase::loadFromDirectory(const ::std::string& directoryPath) {
    if (!fs::exists(directoryPath) || !fs::is_directory(directoryPath)) {
        ::std::cerr << "Directory does not exist: " << directoryPath << ::std::endl;
        return false;
    }

    bool success = true;
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml") {
            if (!loadYamlFile(entry.path().string())) {
                success = false;
            }
        }
    }

    return success;
}

::std::optional<OrbitPattern> PatternDatabase::getPattern(const ::std::string& name) const {
    auto it = _patterns.find(name);
    if (it != _patterns.end()) {
        return it->second;
    }
    return ::std::nullopt;
}

::std::vector<OrbitPattern> PatternDatabase::getPatternsForOrbit(uint32_t orbitId) const {
    auto it = _patternsByOrbit.find(orbitId);
    if (it != _patternsByOrbit.end()) {
        return it->second;
    }
    return {};
}

::std::vector<::std::string> PatternDatabase::getPatternNames() const {
    ::std::vector<::std::string> names;
    names.reserve(_patterns.size());
    for (const auto& pair : _patterns) {
        names.push_back(pair.first);
    }
    return names;
}

bool PatternDatabase::hasPattern(const ::std::string& name) const {
    return _patterns.find(name) != _patterns.end();
}

::std::string PatternDatabase::exportToTableGen(const ::std::string& dialectName) const {
    ::std::string output = "// Auto-generated TableGen for dialect: " + dialectName + "\n\n";

    for (const auto& pair : _patterns) {
        const auto& pattern = pair.second;
        ::std::string sanitizedName = pattern.name;

        // Sanitize name for TableGen identifier
        ::std::replace(sanitizedName.begin(), sanitizedName.end(), '-', '_');
        ::std::replace(sanitizedName.begin(), sanitizedName.end(), '.', '_');

        output += "def " + sanitizedName + " : OrbitPattern {\n";
        output += "  let name = \"" + pattern.name + "\";\n";
        output += "  let orbit_id = " + ::std::to_string(pattern.orbit_id) + "u;\n";

        // Signature patterns
        output += "  let signature_patterns = [";
        for (size_t i = 0; i < pattern.signature_patterns.size(); ++i) {
            if (i > 0) output += ", ";
            output += "\"" + pattern.signature_patterns[i] + "\"";
        }
        output += "];\n";

        output += "  let weight = " + ::std::to_string(pattern.weight) + ";\n";
        output += "}\n\n";
    }

    return output;
}

void PatternDatabase::clear() {
    _patterns.clear();
    _patternsByOrbit.clear();
}

::std::vector<OrbitPattern> PatternDatabase::getPatterns() const {
    ::std::vector<OrbitPattern> patterns;
    patterns.reserve(_patterns.size());
    for (const auto& [name, pattern] : _patterns) {
        patterns.push_back(pattern);
    }
    return patterns;
}

size_t PatternDatabase::size() const {
    return _patterns.size();
}

} // namespace cppfort::ir