#include "semantic_orbit_loader.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

namespace cppfort::stage0 {

bool SemanticOrbitLoader::load_patterns(const std::string& cpp2_path,
                                       const std::string& cpp_path,
                                       const std::string& c_path) {
    patterns_.clear();
    contexts_.clear();
    composition_rules_.clear();

    bool success = true;
    if (!cpp2_path.empty()) success &= parse_yaml_file(cpp2_path, "cpp2");
    if (!cpp_path.empty()) success &= parse_yaml_file(cpp_path, "cpp");
    if (!c_path.empty()) success &= parse_yaml_file(c_path, "c");

    return success;
}

std::vector<SemanticOrbitPattern> SemanticOrbitLoader::get_patterns(const std::string& language) const {
    auto it = patterns_.find(language);
    return it != patterns_.end() ? it->second : std::vector<SemanticOrbitPattern>{};
}

MaskingContext SemanticOrbitLoader::get_context(const std::string& context_name) const {
    auto it = contexts_.find(context_name);
    return it != contexts_.end() ? it->second : MaskingContext{context_name, {}};
}

bool SemanticOrbitLoader::can_contain(const std::string& parent, const std::string& child) const {
    for (const auto& rule : composition_rules_) {
        if (rule.parent == parent) {
            for (const auto& contained : rule.contains) {
                if (contained == child) return true;
            }
        }
    }
    return false;
}

const SemanticOrbitPattern* SemanticOrbitLoader::find_best_pattern(
    const std::string& text,
    const std::string& language,
    const std::string& context) const {
    const auto& lang_patterns = get_patterns(language);
    const auto& ctx = get_context(context);

    const SemanticOrbitPattern* best = nullptr;
    double best_confidence = -1.0;

    for (const auto& pattern : lang_patterns) {
        uint16_t mask = ctx.get_mask(pattern.name);
        double confidence = pattern.compute_confidence(text, mask);
        if (confidence > best_confidence) {
            best_confidence = confidence;
            best = &pattern;
        }
    }

    return best;
}

bool SemanticOrbitLoader::parse_yaml_file(const std::string& path, const std::string& language) {
    try {
        YAML::Node root = YAML::LoadFile(path);

        if (!root.IsMap()) {
            std::cerr << "Error: Root node is not a map in " << path << std::endl;
            return false;
        }

        // Parse patterns
        if (root["patterns"] && root["patterns"].IsSequence()) {
            for (const auto& pattern_node : root["patterns"]) {
                SemanticOrbitPattern pattern;
                pattern.name = pattern_node["name"].as<std::string>();
                pattern.orbit_type = pattern_node["orbit_type"].as<std::string>();
                pattern.masking_ring = pattern_node["masking_ring"].as<uint16_t>(0xFFFF);

                if (pattern_node["evidence"] && pattern_node["evidence"].IsSequence()) {
                    for (const auto& ev_node : pattern_node["evidence"]) {
                        SemanticOrbitPattern::Evidence ev;
                        ev.pattern = ev_node["pattern"].as<std::string>();
                        ev.weight = ev_node["weight"].as<double>(1.0);
                        ev.semantic = ev_node["semantic"].as<std::string>("");
                        pattern.evidence.push_back(ev);
                    }
                }

                if (pattern_node["children"] && pattern_node["children"].IsSequence()) {
                    for (const auto& child : pattern_node["children"]) {
                        pattern.children.push_back(child.as<std::string>());
                    }
                }

                if (pattern_node["metadata"] && pattern_node["metadata"].IsMap()) {
                    for (const auto& meta : pattern_node["metadata"]) {
                        pattern.metadata[meta.first.as<std::string>()] = meta.second.as<std::string>();
                    }
                }

                patterns_[language].push_back(pattern);
            }
        }

        // Parse contexts
        if (root["contexts"] && root["contexts"].IsMap()) {
            for (const auto& ctx_node : root["contexts"]) {
                MaskingContext ctx;
                ctx.context_name = ctx_node.first.as<std::string>();
                if (ctx_node.second.IsMap()) {
                    for (const auto& mask_node : ctx_node.second) {
                        ctx.pattern_masks[mask_node.first.as<std::string>()] = mask_node.second.as<uint16_t>();
                    }
                }
                contexts_[ctx.context_name] = ctx;
            }
        }

        // Parse composition rules
        if (root["composition_rules"] && root["composition_rules"].IsSequence()) {
            for (const auto& rule_node : root["composition_rules"]) {
                CompositionRule rule;
                rule.parent = rule_node["parent"].as<std::string>();
                if (rule_node["contains"] && rule_node["contains"].IsSequence()) {
                    for (const auto& contain : rule_node["contains"]) {
                        rule.contains.push_back(contain.as<std::string>());
                    }
                }
                composition_rules_.push_back(rule);
            }
        }

        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error in " << path << ": " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing " << path << ": " << e.what() << std::endl;
        return false;
    }
}

} // namespace cppfort::stage0