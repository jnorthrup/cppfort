#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pattern_loader.h"

namespace cppfort::stage0 {

enum class UnifiedOrbitCategory {
    Unknown,
    Common,
    COnly,
    CPPOnly,
    CPP2Only
};

struct GrammarTreeNode {
    std::string label;
    UnifiedOrbitCategory category = UnifiedOrbitCategory::Unknown;
    std::vector<const PatternData*> patterns;
    std::map<std::string, std::unique_ptr<GrammarTreeNode>> children;
};

class GrammarTree {
public:
    GrammarTree();

    void clear();
    void insert(const PatternData& pattern);
    const GrammarTreeNode& root() const { return *root_; }

private:
    std::unique_ptr<GrammarTreeNode> root_;

    static UnifiedOrbitCategory classify(const PatternData& pattern);
    GrammarTreeNode* ensure_child(GrammarTreeNode& parent, const std::string& label, UnifiedOrbitCategory category);
};

} // namespace cppfort::stage0

