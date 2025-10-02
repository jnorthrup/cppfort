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

    auto pattern = parsePattern(content);
    if (pattern) {
        _patterns[pattern->name] = *pattern;
        _patternsByOrbit[pattern->orbit_id].push_back(*pattern);
        return true;
    }

    ::std::cerr << "Failed to parse pattern from file: " << filepath << ::std::endl;
    return false;
}

::std::optional<OrbitPattern> PatternDatabase::parsePattern(const ::std::string& yamlContent) {
    // Simple YAML-like parser for orbit patterns
    // Format expected:
    // name: pattern_name
    // orbit_id: 123
    // weight: 0.8
    // signature_patterns:
    //   - pattern1
    //   - pattern2
    // protocol_indicators:
    //   - indicator1
    // version_patterns:
    //   - version1

    OrbitPattern pattern;

    ::std::istringstream iss(yamlContent);
    ::std::string line;
    ::std::string currentSection;

    while (::std::getline(iss, line)) {
        // Remove leading/trailing whitespace
        line.erase(line.begin(), ::std::find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !::std::isspace(ch);
        }));
        line.erase(::std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !::std::isspace(ch);
        }).base(), line.end());

        if (line.empty() || line[0] == '#') continue;

        if (line.find("name:") == 0) {
            pattern.name = line.substr(5);
            pattern.name.erase(pattern.name.begin(), ::std::find_if(pattern.name.begin(), pattern.name.end(), [](unsigned char ch) {
                return !::std::isspace(ch);
            }));
        } else if (line.find("orbit_id:") == 0) {
            ::std::string value = line.substr(9);
            pattern.orbit_id = ::std::stoul(value);
        } else if (line.find("weight:") == 0) {
            ::std::string value = line.substr(7);
            pattern.weight = ::std::stod(value);
        } else if (line.find("signature_patterns:") == 0) {
            currentSection = "signature";
        } else if (line.find("protocol_indicators:") == 0) {
            currentSection = "protocol";
        } else if (line.find("version_patterns:") == 0) {
            currentSection = "version";
        } else if (line[0] == '-' && !currentSection.empty()) {
            ::std::string value = line.substr(1);
            value.erase(value.begin(), ::std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                return !::std::isspace(ch);
            }));

            // Remove surrounding quotes if present
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }

            if (currentSection == "signature") {
                pattern.signature_patterns.push_back(value);
            } else if (currentSection == "protocol") {
                pattern.protocol_indicators.push_back(value);
            } else if (currentSection == "version") {
                pattern.version_patterns.push_back(value);
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