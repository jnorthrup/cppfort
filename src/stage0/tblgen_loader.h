#pragma once

#include <map>
#include <string>
#include <vector>

namespace cppfort::stage0 {

// Semantic unit from tblgen with n-way patterns
struct TblgenSemanticUnit {
    std::string name;
    std::vector<std::string> segments;
    std::string c_pattern;
    std::string cpp_pattern;
    std::string cpp2_pattern;
};

// Load semantic units from tblgen-generated JSON
class TblgenLoader {
public:
    bool load_json(const std::string& path);

    const std::map<std::string, TblgenSemanticUnit>& units() const {
        return units_;
    }

    const TblgenSemanticUnit* get_unit(const std::string& name) const;

private:
    std::map<std::string, TblgenSemanticUnit> units_;
};

} // namespace cppfort::stage0
