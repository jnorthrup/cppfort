#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <cstring>
#include <chrono>
#include <array>
#include <algorithm>

namespace cppfort::crdt {

// Helper: rotate right (for SHA256)
inline constexpr uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

// SHA256 hash wrapper (32 bytes) - pure C++ implementation
struct SHA256Hash {
    std::array<uint8_t, 32> bytes{};

    bool operator==(const SHA256Hash& other) const {
        return bytes == other.bytes;
    }

    bool operator!=(const SHA256Hash& other) const {
        return !(*this == other);
    }

    bool operator<(const SHA256Hash& other) const {
        return bytes < other.bytes;
    }

    bool operator<=(const SHA256Hash& other) const {
        return bytes <= other.bytes;
    }

    bool operator>(const SHA256Hash& other) const {
        return bytes > other.bytes;
    }

    bool operator>=(const SHA256Hash& other) const {
        return bytes >= other.bytes;
    }

    std::string to_hex_string() const {
        constexpr char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (uint8_t b : bytes) {
            result.push_back(hex_chars[b >> 4]);
            result.push_back(hex_chars[b & 0x0f]);
        }
        return result;
    }

    // Pure C++ SHA256 implementation
    static SHA256Hash compute(const std::string& data) {
        return sha256(data);
    }

    static SHA256Hash combine(const std::vector<SHA256Hash>& hashes) {
        if (hashes.empty()) {
            return compute("");
        }
        std::string combined;
        combined.reserve(hashes.size() * 32);
        for (const auto& h : hashes) {
            combined.append(reinterpret_cast<const char*>(h.bytes.data()), 32);
        }
        return compute(combined);
    }

private:
    // SHA256 implementation (from FIPS 180-4)
    static SHA256Hash sha256(const std::string& data) {
        SHA256Hash result;

        // Initialize hash values (first 32 bits of fractional parts of square roots of first 8 primes)
        uint32_t h[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        // Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ae, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        // Pre-processing: add padding
        size_t len = data.size();
        uint64_t bit_len = len * 8;

        std::vector<uint8_t> padded(data.begin(), data.end());
        padded.push_back(0x80);  // Append '1' bit

        // Append '0' bits until length ≡ 448 (mod 512)
        while ((padded.size() * 8) % 512 != 448) {
            padded.push_back(0);
        }

        // Append length as 64-bit big-endian integer
        for (int i = 56; i >= 0; i -= 8) {
            padded.push_back(static_cast<uint8_t>((bit_len >> i) & 0xff));
        }

        // Process message in 512-bit chunks
        for (size_t chunk = 0; chunk < padded.size(); chunk += 64) {
            uint32_t w[64];

            // Copy chunk into first 16 words
            for (int i = 0; i < 16; ++i) {
                w[i] = (static_cast<uint32_t>(padded[chunk + i * 4]) << 24) |
                       (static_cast<uint32_t>(padded[chunk + i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(padded[chunk + i * 4 + 2]) << 8) |
                       static_cast<uint32_t>(padded[chunk + i * 4 + 3]);
            }

            // Extend remaining 48 words
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            // Initialize hash values for this chunk
            uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
            uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];

            // Compression loop
            for (int i = 0; i < 64; ++i) {
                uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                uint32_t ch = (e & f) ^ (~e & g);
                uint32_t temp1 = h_val + S1 + ch + k[i] + w[i];
                uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t temp2 = S0 + maj;

                h_val = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            // Add hash values for this chunk
            h[0] += a; h[1] += b; h[2] += c; h[3] += d;
            h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
        }

        // Produce final hash value
        for (int i = 0; i < 8; ++i) {
            result.bytes[i * 4] = static_cast<uint8_t>((h[i] >> 24) & 0xff);
            result.bytes[i * 4 + 1] = static_cast<uint8_t>((h[i] >> 16) & 0xff);
            result.bytes[i * 4 + 2] = static_cast<uint8_t>((h[i] >> 8) & 0xff);
            result.bytes[i * 4 + 3] = static_cast<uint8_t>(h[i] & 0xff);
        }

        return result;
    }
};

// Unique identifier for AST nodes within a translation unit
using NodeID = uint64_t;

// Semantic hash for an AST node
struct SemanticHash {
    NodeID node_id;                          // Unique ID within translation unit
    SHA256Hash content_hash;                 // Hash of node's semantic content
    SHA256Hash merkle_hash;                  // Hash including children
    std::vector<NodeID> children;            // Child node IDs
    std::string node_kind;                   // e.g., "FunctionDeclaration", "BinaryExpression"

    // Compute Merkle hash from content and children
    void compute_merkle(const std::unordered_map<NodeID, SemanticHash>& all_nodes) {
        std::vector<SHA256Hash> child_hashes;
        child_hashes.reserve(children.size() + 1);
        child_hashes.push_back(content_hash);

        for (NodeID child_id : children) {
            auto it = all_nodes.find(child_id);
            if (it != all_nodes.end()) {
                child_hashes.push_back(it->second.merkle_hash);
            }
        }

        merkle_hash = SHA256Hash::combine(child_hashes);
    }

    // Get full hash path for CRDT addressing
    std::string get_hash_path() const {
        return merkle_hash.to_hex_string();
    }
};

// CRDT patch for AST modifications
struct CRDTPatch {
    enum class Op {
        InsertNode,     // Insert new AST node
        DeleteNode,     // Delete AST node
        UpdateNode,     // Update node content
        MoveNode        // Move node within tree
    };

    Op operation;
    SHA256Hash target_hash;                  // Target node's Merkle hash
    std::optional<SHA256Hash> parent_hash;   // Parent node's hash (for insert/move)
    uint64_t timestamp;
    std::vector<SHA256Hash> dependencies;    // Hashes of depended nodes

    // Patch data (operation-specific)
    std::optional<std::string> node_kind;    // For InsertNode
    std::optional<std::string> content_data; // For InsertNode/UpdateNode
    std::optional<size_t> position;          // For MoveNode

    // Serialize patch for storage/transmission
    std::string serialize() const {
        std::string result;
        result += std::to_string(static_cast<int>(operation));
        result += ":";
        result += target_hash.to_hex_string();
        result += ":";
        result += std::to_string(timestamp);
        return result;
    }
};

// Bidirectional semantic mapping between cpp2 and C++ AST nodes
struct SemanticMapping {
    struct NodeMapping {
        std::string cpp2_kind;           // e.g., "FunctionDeclaration"
        std::string clang_kind;          // e.g., "FunctionDecl"
        std::vector<std::string> cpp2_to_clang_rules;
        std::vector<std::string> clang_to_cpp2_rules;
    };

    static std::unordered_map<std::string, NodeMapping> get_builtin_mappings() {
        return {
            {"FunctionDeclaration", {
                "FunctionDeclaration",
                "FunctionDecl",
                {"name: () -> T", "auto name() -> T"},
                {"auto name() -> T", "name: () -> T"}
            }},
            {"VariableDeclaration", {
                "VariableDeclaration",
                "VarDecl",
                {"x: Type = value", "Type x = value"},
                {"Type x = value", "x: Type = value"}
            }},
            {"Parameter", {
                "Parameter",
                "ParmVarDecl",
                {"inout x: T", "T& x"},
                {"T& x", "inout x: T"}
            }}
        };
    }
};

// Forward declaration visitor interface for computing semantic hashes
class SemanticHashVisitor {
public:
    virtual ~SemanticHashVisitor() = default;

    // Visit AST node and compute its semantic content hash
    virtual SHA256Hash visit(const void* node) = 0;

    // Get the mapping between cpp2 and Clang AST kinds
    virtual std::optional<std::string> map_cpp2_to_clang(const std::string& cpp2_kind) = 0;
    virtual std::optional<std::string> map_clang_to_cpp2(const std::string& clang_kind) = 0;
};

// Semantic hash computation context
class SemanticHashContext {
private:
    std::unordered_map<NodeID, SemanticHash> nodes_;
    NodeID next_id_ = 1;
    std::unique_ptr<SemanticHashVisitor> visitor_;

public:
    explicit SemanticHashContext(std::unique_ptr<SemanticHashVisitor> visitor)
        : visitor_(std::move(visitor)) {}

    // Register a node and compute its hashes
    NodeID register_node(const std::string& node_kind,
                        const std::string& semantic_content,
                        const std::vector<NodeID>& children = {}) {
        SemanticHash hash;
        hash.node_id = next_id_++;
        hash.node_kind = node_kind;
        hash.content_hash = SHA256Hash::compute(semantic_content);
        hash.children = children;
        hash.compute_merkle(nodes_);
        nodes_[hash.node_id] = hash;
        return hash.node_id;
    }

    // Get node by ID
    const SemanticHash* get_node(NodeID id) const {
        auto it = nodes_.find(id);
        return it != nodes_.end() ? &it->second : nullptr;
    }

    // Get node by hash
    const SemanticHash* find_node_by_hash(const SHA256Hash& hash) const {
        for (const auto& [id, node_hash] : nodes_) {
            if (node_hash.merkle_hash == hash) {
                return &node_hash;
            }
        }
        return nullptr;
    }

    // Generate CRDT patch for node insertion
    CRDTPatch create_insert_patch(NodeID parent_id,
                                  const std::string& node_kind,
                                  const std::string& content,
                                  size_t position = 0) {
        CRDTPatch patch;
        patch.operation = CRDTPatch::Op::InsertNode;
        patch.node_kind = node_kind;
        patch.content_data = content;
        patch.position = position;
        patch.timestamp = get_timestamp();

        if (parent_id != 0) {
            const SemanticHash* parent = get_node(parent_id);
            if (parent) {
                patch.parent_hash = parent->merkle_hash;
            }
        }

        return patch;
    }

    // Generate CRDT patch for node deletion
    CRDTPatch create_delete_patch(NodeID target_id) {
        CRDTPatch patch;
        patch.operation = CRDTPatch::Op::DeleteNode;
        patch.timestamp = get_timestamp();

        const SemanticHash* target = get_node(target_id);
        if (target) {
            patch.target_hash = target->merkle_hash;
        }

        return patch;
    }

    // Generate CRDT patch for node update
    CRDTPatch create_update_patch(NodeID target_id, const std::string& new_content) {
        CRDTPatch patch;
        patch.operation = CRDTPatch::Op::UpdateNode;
        patch.content_data = new_content;
        patch.timestamp = get_timestamp();

        const SemanticHash* target = get_node(target_id);
        if (target) {
            patch.target_hash = target->merkle_hash;
        }

        return patch;
    }

    // Apply a patch and return new node ID if applicable
    std::optional<NodeID> apply_patch(const CRDTPatch& patch) {
        switch (patch.operation) {
            case CRDTPatch::Op::InsertNode:
                if (patch.node_kind && patch.content_data) {
                    return register_node(*patch.node_kind, *patch.content_data);
                }
                break;

            case CRDTPatch::Op::DeleteNode: {
                const SemanticHash* target = find_node_by_hash(patch.target_hash);
                if (target) {
                    nodes_.erase(target->node_id);
                    return target->node_id;
                }
                break;
            }

            case CRDTPatch::Op::UpdateNode: {
                const SemanticHash* target = find_node_by_hash(patch.target_hash);
                if (target && patch.content_data) {
                    // Create updated node with same children
                    SemanticHash updated = *target;
                    updated.content_hash = SHA256Hash::compute(*patch.content_data);
                    updated.compute_merkle(nodes_);
                    nodes_[updated.node_id] = updated;
                    return updated.node_id;
                }
                break;
            }

            case CRDTPatch::Op::MoveNode:
                // Move requires re-parenting (not yet implemented)
                break;
        }

        return std::nullopt;
    }

    // Get all nodes
    const std::unordered_map<NodeID, SemanticHash>& get_all_nodes() const {
        return nodes_;
    }

private:
    static uint64_t get_timestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

// Utility for building semantic content strings
struct SemanticContentBuilder {
    std::string content;

    SemanticContentBuilder& add(int value) {
        content += std::to_string(value);
        content += "|";
        return *this;
    }

    SemanticContentBuilder& add(double value) {
        content += std::to_string(value);
        content += "|";
        return *this;
    }

    // Use SFINAE to enable add(uint64_t) only if uint64_t is distinct from size_t
    // or simply merge them if size_t == uint64_t.
    // The simplest fix for "ambiguous overload" if size_t == uint64_t is to NOT declare both.
    // We can use a template to capture integer types.

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, SemanticContentBuilder&>::type
    add(T value) {
        content += std::to_string(value);
        content += "|";
        return *this;
    }

    SemanticContentBuilder& add(const std::string& value) {
        content += value;
        content += "|";
        return *this;
    }

    SemanticContentBuilder& add(const char* value) {
        content += value;
        content += "|";
        return *this;
    }

    SemanticContentBuilder& add(const SHA256Hash& hash) {
        content += hash.to_hex_string();
        content += "|";
        return *this;
    }

    std::string build() const {
        return content;
    }
};

// Factory function to create a cpp2 semantic hash visitor
std::unique_ptr<SemanticHashVisitor> create_cpp2_visitor();

} // namespace cppfort::crdt
