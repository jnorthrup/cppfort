#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>
#include <string>
#include <map>

namespace cppfort::pijul {

constexpr std::size_t HASH_SIZE = 20;
constexpr std::size_t LINE_SIZE = 4;
constexpr std::size_t KEY_SIZE = HASH_SIZE + LINE_SIZE;

using ExternalKey = std::vector<std::uint8_t>;
using ExternalHash = std::vector<std::uint8_t>;

struct Value {
    enum class Kind {
        String
    };

    Kind kind = Kind::String;
    std::string string_value;
};

using AuthorMap = std::map<std::string, Value>;

struct Edge {
    ExternalKey from;
    ExternalKey to;
    ExternalHash introduced_by;
};

struct NewNodesChange {
    std::vector<ExternalKey> up_context;
    std::vector<ExternalKey> down_context;
    std::uint8_t flag = 0;
    std::uint32_t line_num = 0;
    std::vector<std::vector<std::uint8_t>> nodes;
};

struct EdgesChange {
    std::uint8_t flag = 0;
    std::vector<Edge> edges;
};

using Change = std::variant<NewNodesChange, EdgesChange>;

struct Patch {
    std::vector<AuthorMap> authors;
    std::string name;
    std::string description;
    std::string timestamp;
    std::vector<ExternalHash> dependencies;
    std::vector<Change> changes;
};

std::vector<ExternalHash> compute_dependencies(const std::vector<Change>& changes);

Patch make_patch(std::vector<AuthorMap> authors,
                 std::string name,
                 std::string description,
                 std::string timestamp,
                 std::vector<Change> changes);

} // namespace cppfort::pijul
