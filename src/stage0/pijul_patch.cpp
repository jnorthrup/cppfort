#include "pijul_patch.h"

#include <algorithm>

namespace cppfort::pijul {

namespace {

bool is_root_hash(const ExternalHash& hash) {
    if (hash.size() != HASH_SIZE) {
        return false;
    }
    return std::all_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte == 0; });
}

bool contains_hash(const std::vector<ExternalHash>& hashes, const ExternalHash& candidate) {
    return std::any_of(hashes.begin(), hashes.end(), [&](const ExternalHash& current) {
        return current == candidate;
    });
}

void push_dependency(const ExternalHash& candidate, std::vector<ExternalHash>& output) {
    if (candidate.empty() || is_root_hash(candidate)) {
        return;
    }
    if (!contains_hash(output, candidate)) {
        output.push_back(candidate);
    }
}

void extract_from_key(const ExternalKey& key, std::vector<ExternalHash>& output) {
    if (key.size() <= LINE_SIZE) {
        return;
    }
    ExternalHash hash_part(key.begin(), key.end() - LINE_SIZE);
    push_dependency(hash_part, output);
}

} // namespace

std::vector<ExternalHash> compute_dependencies(const std::vector<Change>& changes) {
    std::vector<ExternalHash> dependencies;
    dependencies.reserve(changes.size());

    for (const auto& variant : changes) {
        if (const auto* new_nodes = std::get_if<NewNodesChange>(&variant)) {
            for (const auto& context : new_nodes->up_context) {
                extract_from_key(context, dependencies);
            }
            for (const auto& context : new_nodes->down_context) {
                extract_from_key(context, dependencies);
            }
        } else if (const auto* edges = std::get_if<EdgesChange>(&variant)) {
            for (const auto& edge : edges->edges) {
                extract_from_key(edge.from, dependencies);
                extract_from_key(edge.to, dependencies);
                if (!edge.introduced_by.empty()) {
                    push_dependency(edge.introduced_by, dependencies);
                }
            }
        }
    }

    return dependencies;
}

Patch make_patch(std::vector<AuthorMap> authors,
                 std::string name,
                 std::string description,
                 std::string timestamp,
                 std::vector<Change> changes) {
    Patch patch;
    patch.authors = std::move(authors);
    patch.name = std::move(name);
    patch.description = std::move(description);
    patch.timestamp = std::move(timestamp);
    patch.dependencies = compute_dependencies(changes);
    patch.changes = std::move(changes);
    return patch;
}

} // namespace cppfort::pijul
