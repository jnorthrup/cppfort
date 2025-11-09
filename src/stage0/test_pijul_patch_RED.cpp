#include "pijul_patch.h"
#include "heap_limiter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace cppfort::pijul;

namespace {

ExternalHash make_hash(std::uint8_t fill) {
    return ExternalHash(HASH_SIZE, fill);
}

ExternalKey make_key(std::uint8_t fill, bool append_line = true) {
    ExternalKey key(HASH_SIZE, fill);
    if (append_line) {
        key.push_back(0x01);
        key.push_back(0x00);
        key.push_back(0x00);
        key.push_back(0x00);
    }
    return key;
}

ExternalKey make_root_key() {
    return ExternalKey(KEY_SIZE, 0);
}

bool contains_hash(const std::vector<ExternalHash>& hashes, const ExternalHash& target) {
    return std::any_of(hashes.begin(), hashes.end(), [&](const ExternalHash& current) {
        return current == target;
    });
}

std::size_t unique_hash_count(const std::vector<ExternalHash>& hashes) {
    std::vector<ExternalHash> copy = hashes;
    std::sort(copy.begin(), copy.end());
    copy.erase(std::unique(copy.begin(), copy.end()), copy.end());
    return copy.size();
}

} // namespace

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    NewNodesChange new_nodes;
    new_nodes.up_context.push_back(make_key(0xAA));
    new_nodes.up_context.push_back(make_root_key());
    new_nodes.down_context.push_back(make_key(0xAB));
    new_nodes.nodes.push_back(std::vector<std::uint8_t>{'n', 'o', 'd', 'e'});

    EdgesChange edges;
    Edge e1;
    e1.from = make_key(0xBB);
    e1.to = make_key(0xCC);
    e1.introduced_by = make_hash(0xDD);
    edges.edges.push_back(e1);

    Edge e2;
    e2.from = make_key(0xBB);
    e2.to = make_key(0xCC);
    edges.edges.push_back(e2);

    std::vector<Change> changes;
    changes.emplace_back(new_nodes);
    changes.emplace_back(edges);

    const auto deps = compute_dependencies(changes);

    bool ok = true;
    if (deps.size() != 5) {
        std::cerr << "Expected 5 dependency hashes, got " << deps.size() << "\n";
        ok = false;
    }
    if (unique_hash_count(deps) != deps.size()) {
        std::cerr << "Dependencies are not unique\n";
        ok = false;
    }
    if (!contains_hash(deps, make_hash(0xAA))) {
        std::cerr << "Missing dependency for up_context hash 0xAA\n";
        ok = false;
    }
    if (!contains_hash(deps, make_hash(0xAB))) {
        std::cerr << "Missing dependency for down_context hash 0xAB\n";
        ok = false;
    }
    if (!contains_hash(deps, make_hash(0xBB))) {
        std::cerr << "Missing dependency for edge.from hash 0xBB\n";
        ok = false;
    }
    if (!contains_hash(deps, make_hash(0xCC))) {
        std::cerr << "Missing dependency for edge.to hash 0xCC\n";
        ok = false;
    }
    if (!contains_hash(deps, make_hash(0xDD))) {
        std::cerr << "Missing dependency for introduced_by hash 0xDD\n";
        ok = false;
    }

    if (contains_hash(deps, make_hash(0x00))) {
        std::cerr << "Unexpected ROOT_KEY dependency found\n";
        ok = false;
    }

    std::vector<AuthorMap> authors;
    AuthorMap author;
    Value name_value;
    name_value.kind = Value::Kind::String;
    name_value.string_value = "codex";
    author.emplace("name", name_value);
    authors.push_back(author);

    Patch patch = make_patch(authors, "unit-test", "diff", "2025-11-04T00:00:00Z", changes);
    if (patch.dependencies != deps) {
        std::cerr << "Patch dependencies differ from compute_dependencies\n";
        ok = false;
    }
    if (patch.changes.size() != changes.size()) {
        std::cerr << "Patch did not retain change list\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
