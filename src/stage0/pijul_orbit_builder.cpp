#include "pijul_orbit_builder.h"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace cppfort {
namespace pijul {

namespace {

std::string format_hierarchy_signature(const NodeContext& ctx);
std::string annotate_fragment(const NodeContext& ctx, const std::string& fragment);

Edge make_edge(const std::string& from,
               const std::string& to,
               const std::string& introduced_by,
               const NodeContext& context) {
    Edge edge{from, to, introduced_by};
    const std::string signature = format_hierarchy_signature(context);
    if (!signature.empty()) {
        edge.introduced_by = introduced_by + "@" + signature;
    }
    return edge;
}

std::string extract_fragment(std::string_view text, const NodeContext& ctx) {
    if (ctx.end_pos <= ctx.start_pos || ctx.end_pos > text.size()) {
        return {};
    }
    return std::string(text.substr(ctx.start_pos, ctx.end_pos - ctx.start_pos));
}

void append_unique(std::vector<std::string>& container, const std::string& value) {
    if (value.empty()) {
        return;
    }
    if (std::find(container.begin(), container.end(), value) == container.end()) {
        container.push_back(value);
    }
}

std::string format_hierarchy_signature(const NodeContext& ctx) {
    if (ctx.hierarchy_index.empty() && ctx.hierarchy_label.empty()) {
        return {};
    }
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < ctx.hierarchy_index.size(); ++i) {
        if (i != 0) {
            oss << '.';
        }
        oss << ctx.hierarchy_index[i];
    }
    oss << "]";
    if (!ctx.hierarchy_label.empty()) {
        oss << ctx.hierarchy_label;
    }
    return oss.str();
}

std::string annotate_fragment(const NodeContext& ctx, const std::string& fragment) {
    const std::string signature = format_hierarchy_signature(ctx);
    if (signature.empty()) {
        return fragment;
    }
    if (fragment.empty()) {
        return signature;
    }
    return signature + " " + fragment;
}

NodeContext enrich_context(const NodeContext& base,
                           const std::string& key,
                           const std::vector<std::string>& up_extra,
                           const std::vector<std::string>& down_extra) {
    NodeContext ctx = base;
    if (ctx.orbit_signature.empty()) {
        ctx.orbit_signature = key;
    }
    for (const auto& entry : up_extra) {
        append_unique(ctx.up_context, entry);
    }
    for (const auto& entry : down_extra) {
        append_unique(ctx.down_context, entry);
    }
    return ctx;
}

std::string lookup_rewrite(const std::unordered_map<std::string, std::string>* rewrites,
                           const NodeContext& ctx,
                           const std::string& fallback) {
    if (!rewrites) {
        return fallback;
    }
    const std::string signature = format_hierarchy_signature(ctx);
    auto it = rewrites->find(signature);
    if (it != rewrites->end()) {
        return it->second;
    }
    return fallback;
}

} // namespace

void build_orbit_patch(const OrbitMatchCollection& source,
                       const OrbitMatchCollection& transformed,
                       const std::string& source_code,
                       const std::string& transformed_code,
                       const std::string& patchName,
                       Patch& patch,
                       const std::unordered_map<std::string, std::string>* sourceRewrites,
                       const std::unordered_map<std::string, std::string>* transformedRewrites) {
    (void)source_code;
    std::vector<std::string> added;
    std::vector<std::string> removed;

    for (const auto& [key, info] : transformed.byKey) {
        (void)info;
        if (source.byKey.find(key) == source.byKey.end()) {
            added.push_back(key);
        }
    }

    for (const auto& [key, info] : source.byKey) {
        (void)info;
        if (transformed.byKey.find(key) == transformed.byKey.end()) {
            removed.push_back(key);
        }
    }

    std::unordered_set<std::string> matchedSources;
    std::unordered_set<std::string> addedKeys(added.begin(), added.end());

    for (const auto& key : added) {
        const auto& transformedInfo = transformed.byKey.at(key);
        const std::string fragment = lookup_rewrite(
            transformedRewrites,
            transformedInfo.context,
            extract_fragment(transformed_code, transformedInfo.context));

        NewNodesChange newNode;
        const std::string payload = fragment.empty() ? key : fragment;
        newNode.nodes.push_back(annotate_fragment(transformedInfo.context, payload));
        newNode.context = enrich_context(transformedInfo.context, key, {}, {});
        patch.changes.emplace_back(std::move(newNode));

        auto patternIt = source.byPattern.find(transformedInfo.patternName);
        if (patternIt == source.byPattern.end()) {
            continue;
        }

        for (const auto& candidate : patternIt->second) {
            if (!matchedSources.insert(candidate).second) {
                continue;
            }
            const auto& sourceInfo = source.byKey.at(candidate);
            EdgesChange edgeChange;
            edgeChange.context = enrich_context(sourceInfo.context, candidate, {}, {key});
            edgeChange.edges.push_back(make_edge(candidate, key, patchName, sourceInfo.context));
            patch.changes.emplace_back(std::move(edgeChange));
            break;
        }
    }

    for (const auto& key : removed) {
        if (matchedSources.find(key) != matchedSources.end()) {
            continue;
        }
        const auto& sourceInfo = source.byKey.at(key);
        // Represent removal via edge pointing to empty target with contextual metadata.
        EdgesChange removalChange;
        removalChange.context = enrich_context(sourceInfo.context, key, {}, {});
        removalChange.edges.push_back(make_edge(key, "", patchName, sourceInfo.context));
        patch.changes.emplace_back(std::move(removalChange));
    }

    for (const auto& [key, info] : transformed.byKey) {
        if (addedKeys.find(key) != addedKeys.end()) {
            continue;
        }
        if (source.byKey.find(key) == source.byKey.end()) {
            continue;
        }

        EdgesChange identityChange;
        identityChange.context = enrich_context(info.context, key, {}, {});
        identityChange.edges.push_back(make_edge(key, key, patchName, info.context));
        patch.changes.emplace_back(std::move(identityChange));
    }
}

} // namespace pijul
} // namespace cppfort
