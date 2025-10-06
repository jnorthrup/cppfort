#include "grammar_tree.h"

#include <algorithm>
#include <cctype>

namespace cppfort::stage0 {
namespace {

bool contains_case_insensitive(const std::string& text, const std::string& needle) {
    auto it = std::search(text.begin(), text.end(), needle.begin(), needle.end(),
                          [](char a, char b) {
                              return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
                          });
    return it != text.end();
}

} // namespace

GrammarTree::GrammarTree()
    : root_(std::make_unique<GrammarTreeNode>()) {
    root_->label = "root";
    root_->category = UnifiedOrbitCategory::Common;
}

void GrammarTree::clear() {
    root_ = std::make_unique<GrammarTreeNode>();
    root_->label = "root";
    root_->category = UnifiedOrbitCategory::Common;
}

UnifiedOrbitCategory GrammarTree::classify(const PatternData& pattern) {
    const auto lower_category = [](const std::string& value) {
        std::string lowered;
        lowered.resize(value.size());
        std::transform(value.begin(), value.end(), lowered.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return lowered;
    };

    std::string category = lower_category(pattern.category);
    if (category.find("cpp2") != std::string::npos) {
        return UnifiedOrbitCategory::CPP2Only;
    }
    if (category.find("cpp") != std::string::npos) {
        return UnifiedOrbitCategory::CPPOnly;
    }
    if (category.find("c-only") != std::string::npos || category == "c") {
        return UnifiedOrbitCategory::COnly;
    }

    // Fallback: look at name/signatures for hints
    if (contains_case_insensitive(pattern.name, "inspect") ||
        contains_case_insensitive(pattern.name, "contract")) {
        return UnifiedOrbitCategory::CPP2Only;
    }
    if (contains_case_insensitive(pattern.name, "template") ||
        contains_case_insensitive(pattern.name, "namespace")) {
        return UnifiedOrbitCategory::CPPOnly;
    }

    return UnifiedOrbitCategory::Common;
}

GrammarTreeNode* GrammarTree::ensure_child(GrammarTreeNode& parent, const std::string& label, UnifiedOrbitCategory category) {
    auto it = parent.children.find(label);
    if (it == parent.children.end()) {
        auto node = std::make_unique<GrammarTreeNode>();
        node->label = label;
        node->category = category == UnifiedOrbitCategory::Unknown ? parent.category : category;
        it = parent.children.emplace(label, std::move(node)).first;
    }
    return it->second.get();
}

void GrammarTree::insert(const PatternData& pattern) {
    auto category = classify(pattern);
    GrammarTreeNode* current = root_.get();

    // Branch by category first
    std::string branch_label;
    switch (category) {
        case UnifiedOrbitCategory::COnly: branch_label = "C"; break;
        case UnifiedOrbitCategory::CPPOnly: branch_label = "CPP"; break;
        case UnifiedOrbitCategory::CPP2Only: branch_label = "CPP2"; break;
        case UnifiedOrbitCategory::Common:
        case UnifiedOrbitCategory::Unknown:
        default:
            branch_label = "Common";
            break;
    }

    current = ensure_child(*current, branch_label, category == UnifiedOrbitCategory::Common ? UnifiedOrbitCategory::Common : category);

    // Shared trunk node for similar patterns via name prefix
    std::string trunk_label = pattern.name.substr(0, std::min<size_t>(pattern.name.size(), 3));
    current = ensure_child(*current, trunk_label, category);

    current->patterns.push_back(&pattern);
}

} // namespace cppfort::stage0

