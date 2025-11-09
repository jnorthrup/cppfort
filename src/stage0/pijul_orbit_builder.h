#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "pijul_types.h"

namespace cppfort {
namespace pijul {

struct OrbitMatchInfo {
    std::string key;
    std::string patternName;
    NodeContext context;
};

struct OrbitMatchCollection {
    std::unordered_map<std::string, OrbitMatchInfo> byKey;
    std::unordered_map<std::string, std::vector<std::string>> byPattern;
};

void build_orbit_patch(const OrbitMatchCollection& source,
                       const OrbitMatchCollection& transformed,
                       const std::string& source_code,
                       const std::string& transformed_code,
                       const std::string& patchName,
                       Patch& patch,
                       const std::unordered_map<std::string, std::string>* sourceRewrites = nullptr,
                       const std::unordered_map<std::string, std::string>* transformedRewrites = nullptr);

} // namespace pijul
} // namespace cppfort
