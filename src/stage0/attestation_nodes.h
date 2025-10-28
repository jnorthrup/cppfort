#pragma once

#include "node.h"
#include <string>
#include <vector>
#include <array>

namespace cppfort::ir {

/**
 * Attestation & Anti-Cheat Node Infrastructure
 *
 * These nodes embed cryptographic checkpoints in the IR graph to enable:
 * - Deterministic compilation verification
 * - Merkle tree proof of transformation sequences
 * - Code injection detection
 * - Binary tampering prevention
 */

// ============================================================================
// SHA3-512 Hash Type (64 bytes)
// ============================================================================
using Hash512 = std::array<uint8_t, 64>;

// ============================================================================
// Ed25519 Signature Type (64 bytes)
// ============================================================================
using Signature = std::array<uint8_t, 64>;

/**
 * MerkleCheckpointNode - Attestation anchor point in transformation pipeline
 *
 * Records the hash of a subgraph at a specific compilation stage.
 * Forms merkle tree by linking parent checkpoints.
 *
 * Input 0: Previous checkpoint (or nullptr for root)
 * Input 1+: All nodes included in this checkpoint's hash
 */
class MerkleCheckpointNode : public Node {
private:
    Hash512 _hash;           // SHA3-512 of all input nodes
    std::string _stageName;  // "CPP2->IR", "IR->Optimize", "IR->C++", etc
    uint64_t _timestamp;     // Deterministic build timestamp

public:
    MerkleCheckpointNode(Node* parent, const std::string& stage)
        : Node(), _stageName(stage), _timestamp(0) {
        setInput(0, parent);
        _hash = {}; // Computed lazily on first access
    }

    std::string label() const override { return "Merkle[" + _stageName + "]"; }
    NodeKind getKind() const override { return NodeKind::MERKLE_CHECKPOINT; }

    const Hash512& hash();  // Lazy computation
    const std::string& stageName() const { return _stageName; }
    uint64_t timestamp() const { return _timestamp; }
    void setTimestamp(uint64_t ts) { _timestamp = ts; }

    // Add a node to this checkpoint's coverage
    void addCoveredNode(Node* n) {
        _inputs.push_back(n);
        _hash = {};  // Invalidate cached hash
    }

    Type* compute() override;
    Node* peephole() override { return nullptr; }  // Never optimize away
};

/**
 * HashWitnessNode - Embeds a hash value as an IR constant
 *
 * Used to encode expected hashes for verification.
 * Cannot be optimized away or dead-code eliminated.
 */
class HashWitnessNode : public Node {
private:
    Hash512 _expectedHash;
    std::string _description;

public:
    HashWitnessNode(const Hash512& hash, const std::string& desc)
        : Node(), _expectedHash(hash), _description(desc) {}

    std::string label() const override { return "HashWitness[" + _description + "]"; }
    NodeKind getKind() const override { return NodeKind::HASH_WITNESS; }

    const Hash512& expectedHash() const { return _expectedHash; }
    bool hasSideEffects() const override { return true; }  // Prevent elimination

    Type* compute() override;
    Node* peephole() override { return nullptr; }
};

/**
 * SignaturePointNode - Cryptographic signature anchor
 *
 * Marks points where Ed25519 signatures are verified.
 * Used to prove compiler binary integrity and source authenticity.
 */
class SignaturePointNode : public Node {
private:
    Signature _signature;
    std::array<uint8_t, 32> _publicKey;  // Ed25519 public key
    std::string _signedArtifact;         // "compiler_binary", "source_hash", etc

public:
    SignaturePointNode(const Signature& sig, const std::array<uint8_t, 32>& pubkey,
                       const std::string& artifact)
        : Node(), _signature(sig), _publicKey(pubkey), _signedArtifact(artifact) {}

    std::string label() const override { return "Signature[" + _signedArtifact + "]"; }
    NodeKind getKind() const override { return NodeKind::SIGNATURE_POINT; }

    const Signature& signature() const { return _signature; }
    const std::array<uint8_t, 32>& publicKey() const { return _publicKey; }
    bool hasSideEffects() const override { return true; }

    Type* compute() override;
    Node* peephole() override { return nullptr; }
};

/**
 * TamperGuardNode - Runtime anti-tamper check
 *
 * Inserts code that verifies node graph structure hasn't been modified.
 * Detects unexpected nodes, missing edges, altered types.
 *
 * Input 0: Subgraph root to check
 */
class TamperGuardNode : public CFGNode {
private:
    Hash512 _expectedGraphHash;
    std::string _failureAction;  // "abort", "log", "degrade"

public:
    TamperGuardNode(Node* subgraphRoot, const Hash512& expectedHash,
                    const std::string& action = "abort")
        : CFGNode(), _expectedGraphHash(expectedHash), _failureAction(action) {
        setInput(0, subgraphRoot);
    }

    std::string label() const override { return "TamperGuard"; }
    NodeKind getKind() const override { return NodeKind::TAMPER_GUARD; }

    const Hash512& expectedHash() const { return _expectedGraphHash; }
    bool hasSideEffects() const override { return true; }
    bool blockHead() const override { return true; }

    Type* compute() override;
    Node* peephole() override { return nullptr; }
};

/**
 * InjectionBarrierNode - Prevents code injection attacks
 *
 * Validates that all nodes in a region originated from legitimate parser.
 * Checks for foreign symbols, unexpected IR patterns, AST diff violations.
 *
 * Input 0: Control
 * Input 1+: All nodes in protected region
 */
class InjectionBarrierNode : public CFGNode {
private:
    std::vector<std::string> _allowedSymbols;
    bool _strictMode;

public:
    InjectionBarrierNode(Node* ctrl, const std::vector<std::string>& symbols, bool strict = true)
        : CFGNode(), _allowedSymbols(symbols), _strictMode(strict) {
        setInput(0, ctrl);
    }

    std::string label() const override { return "InjectionBarrier"; }
    NodeKind getKind() const override { return NodeKind::INJECTION_BARRIER; }

    const std::vector<std::string>& allowedSymbols() const { return _allowedSymbols; }
    bool isStrict() const { return _strictMode; }
    bool hasSideEffects() const override { return true; }

    Type* compute() override;
    Node* peephole() override { return nullptr; }
};

/**
 * DeterminismFenceNode - Enforces deterministic compilation ordering
 *
 * Prevents non-deterministic optimizations that would break attestation.
 * Forces stable ordering of parallel compilation units.
 *
 * Input 0: Control
 * Input 1+: Nodes that must be ordered deterministically
 */
class DeterminismFenceNode : public CFGNode {
private:
    uint64_t _sequenceNumber;  // Stable sort key
    std::string _barrierName;

public:
    DeterminismFenceNode(Node* ctrl, uint64_t sequence, const std::string& name)
        : CFGNode(), _sequenceNumber(sequence), _barrierName(name) {
        setInput(0, ctrl);
    }

    std::string label() const override { return "DeterminismFence[" + std::to_string(_sequenceNumber) + "]"; }
    NodeKind getKind() const override { return NodeKind::DETERMINISM_FENCE; }

    uint64_t sequence() const { return _sequenceNumber; }
    bool hasSideEffects() const override { return true; }
    bool blockHead() const override { return true; }

    Type* compute() override;
    Node* peephole() override { return nullptr; }
};

} // namespace cppfort::ir
