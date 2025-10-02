#include "attestation_nodes.h"
#include "type.h"
#include <cstring>
#include <functional>

namespace cppfort::ir {

// ============================================================================
// SHA3-512 Hash Computation (simplified - real impl would use crypto library)
// ============================================================================
static Hash512 computeSHA3_512(const std::vector<Node*>& nodes) {
    Hash512 hash = {};

    // Simplified deterministic hash: hash node IDs + types
    // Production would use actual SHA3-512 from OpenSSL/libsodium
    std::hash<std::string> hasher;
    size_t combined = 0;

    for (const auto* node : nodes) {
        if (!node) continue;
        combined ^= hasher(std::to_string(node->id()));
        combined ^= hasher(node->label());
        if (node->_type) {
            combined ^= hasher(node->_type->toString());
        }
    }

    // Fill hash with deterministic bytes
    memcpy(hash.data(), &combined, std::min(sizeof(combined), hash.size()));

    return hash;
}

// ============================================================================
// MerkleCheckpointNode Implementation
// ============================================================================
const Hash512& MerkleCheckpointNode::hash() {
    // Lazy computation - only compute if not cached
    bool isZero = true;
    for (auto byte : _hash) {
        if (byte != 0) {
            isZero = false;
            break;
        }
    }

    if (isZero && !_inputs.empty()) {
        _hash = computeSHA3_512(_inputs);
    }

    return _hash;
}

Type* MerkleCheckpointNode::compute() {
    // Merkle checkpoints have TOP type - they exist for side effects only
    return Type::TOP;
}

// ============================================================================
// HashWitnessNode Implementation
// ============================================================================
Type* HashWitnessNode::compute() {
    // Hash witnesses have TOP type (array of bytes)
    return Type::TOP;
}

// ============================================================================
// SignaturePointNode Implementation
// ============================================================================
Type* SignaturePointNode::compute() {
    // Signature points have integer type (verification success/failure)
    return TypeInteger::constant(1);  // Boolean result
}

// ============================================================================
// TamperGuardNode Implementation
// ============================================================================
Type* TamperGuardNode::compute() {
    // Tamper guards produce control flow - use TOP
    return Type::TOP;
}

// ============================================================================
// InjectionBarrierNode Implementation
// ============================================================================
Type* InjectionBarrierNode::compute() {
    // Injection barriers produce control flow - use TOP
    return Type::TOP;
}

// ============================================================================
// DeterminismFenceNode Implementation
// ============================================================================
Type* DeterminismFenceNode::compute() {
    // Determinism fences produce control flow - use TOP
    return Type::TOP;
}

} // namespace cppfort::ir
