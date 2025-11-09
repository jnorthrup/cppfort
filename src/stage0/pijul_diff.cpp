#include "pijul_diff.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

namespace cppfort::pijul {

namespace {

constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

ExternalHash hash_line(const std::string& line) {
    ExternalHash hash(HASH_SIZE, 0);

    auto fnv = [](std::uint64_t seed, std::string_view data) {
        std::uint64_t value = seed;
        for (unsigned char ch : data) {
            value ^= ch;
            value *= FNV_PRIME;
        }
        return value;
    };

    std::uint64_t h1 = fnv(FNV_OFFSET, line);
    std::uint64_t h2 = fnv(h1 ^ 0x9e3779b185ebca87ULL, line);
    std::uint64_t h3 = fnv(h2 ^ 0xbf58476d1ce4e5b9ULL, line);

    for (int i = 0; i < 8; ++i) {
        hash[i] = static_cast<std::uint8_t>((h1 >> (i * 8)) & 0xFF);
    }
    for (int i = 0; i < 8; ++i) {
        hash[8 + i] = static_cast<std::uint8_t>((h2 >> (i * 8)) & 0xFF);
    }
    for (int i = 0; i < 4; ++i) {
        hash[16 + i] = static_cast<std::uint8_t>((h3 >> (i * 8)) & 0xFF);
    }
    return hash;
}

ExternalKey make_key(const std::string& line, std::uint32_t ordinal) {
    ExternalKey key = hash_line(line);
    key.push_back(static_cast<std::uint8_t>(ordinal & 0xFF));
    key.push_back(static_cast<std::uint8_t>((ordinal >> 8) & 0xFF));
    key.push_back(static_cast<std::uint8_t>((ordinal >> 16) & 0xFF));
    key.push_back(static_cast<std::uint8_t>((ordinal >> 24) & 0xFF));
    return key;
}

ExternalKey root_key() {
    return ExternalKey(KEY_SIZE, 0);
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::string current;
    current.reserve(64);

    for (char ch : text) {
        current.push_back(ch);
        if (ch == '\n') {
            lines.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

std::vector<ExternalKey> build_keys(const std::vector<std::string>& lines) {
    std::vector<ExternalKey> keys;
    keys.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        keys.push_back(make_key(lines[i], static_cast<std::uint32_t>(i)));
    }
    return keys;
}

} // namespace

Patch compute_line_patch(const std::string& patch_name,
                         const std::string& source,
                         const std::string& target) {
    auto source_lines = split_lines(source);
    auto target_lines = split_lines(target);

    auto source_keys = build_keys(source_lines);
    auto target_keys = build_keys(target_lines);

    std::size_t prefix = 0;
    while (prefix < source_lines.size() &&
           prefix < target_lines.size() &&
           source_lines[prefix] == target_lines[prefix]) {
        ++prefix;
    }

    std::size_t suffix = 0;
    while (suffix < (source_lines.size() - prefix) &&
           suffix < (target_lines.size() - prefix) &&
           source_lines[source_lines.size() - 1 - suffix] ==
               target_lines[target_lines.size() - 1 - suffix]) {
        ++suffix;
    }

    std::vector<Change> changes;

    // Collect inserted lines (target)
    std::vector<std::size_t> inserted_indices;
    for (std::size_t i = prefix; i < target_lines.size() - suffix; ++i) {
        inserted_indices.push_back(i);
    }

    if (!inserted_indices.empty()) {
        NewNodesChange new_nodes;
        new_nodes.flag = 0;
        new_nodes.line_num = static_cast<std::uint32_t>(prefix);

        ExternalKey up_context_key =
            (prefix > 0) ? target_keys[prefix - 1] : root_key();
        ExternalKey down_context_key =
            (prefix + inserted_indices.size() < target_keys.size())
                ? target_keys[prefix + inserted_indices.size()]
                : root_key();

        new_nodes.up_context.push_back(up_context_key);
        new_nodes.down_context.push_back(down_context_key);

        std::vector<ExternalKey> new_keys;
        new_keys.reserve(inserted_indices.size());

        for (std::size_t idx : inserted_indices) {
            const auto& line = target_lines[idx];
            new_nodes.nodes.emplace_back(line.begin(), line.end());
            new_keys.push_back(target_keys[idx]);
        }

        changes.emplace_back(std::move(new_nodes));

        EdgesChange edge_change;
        edge_change.flag = 0;

        ExternalHash intro_hash;

        intro_hash = hash_line(target_lines[inserted_indices.front()]);
        edge_change.edges.push_back(
            Edge{up_context_key, new_keys.front(), intro_hash});

        for (std::size_t i = 1; i < new_keys.size(); ++i) {
            intro_hash = hash_line(target_lines[inserted_indices[i]]);
            edge_change.edges.push_back(
                Edge{new_keys[i - 1], new_keys[i], intro_hash});
        }

        intro_hash = hash_line(target_lines[inserted_indices.back()]);
        edge_change.edges.push_back(
            Edge{new_keys.back(), down_context_key, intro_hash});

        if (!edge_change.edges.empty()) {
            changes.emplace_back(std::move(edge_change));
        }
    }

    return make_patch({}, patch_name, "", "", std::move(changes));
}

} // namespace cppfort::pijul
