#include "attestation_nodes.h"
#include "node.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_merkle_checkpoint() {
    std::cout << "=== Testing MerkleCheckpointNode ===" << std::endl;

    // Create a simple computation graph
    auto* c1 = new ConstantNode(42, nullptr);
    auto* c2 = new ConstantNode(100, nullptr);
    auto* add = new AddNode(c1, c2);

    // Create merkle checkpoint covering these nodes
    auto* checkpoint = new MerkleCheckpointNode(nullptr, "Test Stage");
    checkpoint->addCoveredNode(c1);
    checkpoint->addCoveredNode(c2);
    checkpoint->addCoveredNode(add);

    // Get hash (triggers lazy computation)
    const Hash512& hash1 = checkpoint->hash();

    // Verify hash is non-zero
    bool hasNonZero = false;
    for (auto byte : hash1) {
        if (byte != 0) {
            hasNonZero = true;
            break;
        }
    }
    assert(hasNonZero && "Hash should be non-zero");

    // Same nodes should produce same hash
    auto* checkpoint2 = new MerkleCheckpointNode(nullptr, "Test Stage");
    checkpoint2->addCoveredNode(c1);
    checkpoint2->addCoveredNode(c2);
    checkpoint2->addCoveredNode(add);
    const Hash512& hash2 = checkpoint2->hash();

    assert(hash1 == hash2 && "Deterministic hashing failed");

    std::cout << "✓ Merkle checkpoints produce deterministic hashes" << std::endl;

    delete checkpoint;
    delete checkpoint2;
    delete add;
    delete c2;
    delete c1;
}

void test_hash_witness() {
    std::cout << "=== Testing HashWitnessNode ===" << std::endl;

    Hash512 expectedHash = {};
    expectedHash[0] = 0xDE;
    expectedHash[1] = 0xAD;
    expectedHash[2] = 0xBE;
    expectedHash[3] = 0xEF;

    auto* witness = new HashWitnessNode(expectedHash, "Source Hash");

    assert(witness->expectedHash() == expectedHash);
    assert(witness->hasSideEffects() && "Hash witnesses must not be optimized away");
    assert(witness->getKind() == NodeKind::HASH_WITNESS);

    std::cout << "✓ Hash witnesses preserve expected hashes" << std::endl;

    delete witness;
}

void test_signature_point() {
    std::cout << "=== Testing SignaturePointNode ===" << std::endl;

    Signature sig = {};
    sig[0] = 0xCA;
    sig[1] = 0xFE;

    std::array<uint8_t, 32> pubkey = {};
    pubkey[0] = 0xBA;
    pubkey[1] = 0xBE;

    auto* sigPoint = new SignaturePointNode(sig, pubkey, "Compiler Binary");

    assert(sigPoint->signature() == sig);
    assert(sigPoint->publicKey() == pubkey);
    assert(sigPoint->hasSideEffects() && "Signatures must not be optimized away");
    assert(sigPoint->getKind() == NodeKind::SIGNATURE_POINT);

    std::cout << "✓ Signature points preserve cryptographic data" << std::endl;

    delete sigPoint;
}

void test_tamper_guard() {
    std::cout << "=== Testing TamperGuardNode ===" << std::endl;

    auto* root = new ConstantNode(123, nullptr);
    Hash512 expectedHash = {};
    expectedHash[0] = 0x42;

    auto* guard = new TamperGuardNode(root, expectedHash, "abort");

    assert(guard->expectedHash() == expectedHash);
    assert(guard->hasSideEffects());
    assert(guard->blockHead() && "Tamper guards should be basic block heads");
    assert(guard->getKind() == NodeKind::TAMPER_GUARD);

    std::cout << "✓ Tamper guards protect subgraphs" << std::endl;

    delete guard;
    delete root;
}

void test_injection_barrier() {
    std::cout << "=== Testing InjectionBarrierNode ===" << std::endl;

    auto* ctrl = new ConstantNode(0, nullptr);
    std::vector<std::string> allowed = {"std::cout", "main", "foo"};

    auto* barrier = new InjectionBarrierNode(ctrl, allowed, true);

    assert(barrier->allowedSymbols() == allowed);
    assert(barrier->isStrict());
    assert(barrier->hasSideEffects());
    assert(barrier->getKind() == NodeKind::INJECTION_BARRIER);

    std::cout << "✓ Injection barriers validate symbol whitelist" << std::endl;

    delete barrier;
    delete ctrl;
}

void test_determinism_fence() {
    std::cout << "=== Testing DeterminismFenceNode ===" << std::endl;

    auto* ctrl = new ConstantNode(0, nullptr);
    auto* fence = new DeterminismFenceNode(ctrl, 12345, "Parallel Barrier");

    assert(fence->sequence() == 12345);
    assert(fence->hasSideEffects());
    assert(fence->blockHead());
    assert(fence->getKind() == NodeKind::DETERMINISM_FENCE);

    std::cout << "✓ Determinism fences enforce stable ordering" << std::endl;

    delete fence;
    delete ctrl;
}

void test_attestation_integration() {
    std::cout << "=== Testing Attestation Integration ===" << std::endl;

    // Simulate a compilation pipeline with attestation checkpoints

    // Stage 1: CPP2 → IR
    auto* srcNode = new ConstantNode(1, nullptr);
    auto* checkpoint1 = new MerkleCheckpointNode(nullptr, "CPP2->IR");
    checkpoint1->addCoveredNode(srcNode);

    // Stage 2: IR → Optimize
    auto* optNode = new AddNode(srcNode, new ConstantNode(1, nullptr));
    auto* checkpoint2 = new MerkleCheckpointNode(checkpoint1, "IR->Optimize");
    checkpoint2->addCoveredNode(optNode);

    // Stage 3: Optimize → C++
    auto* emitNode = new ConstantNode(42, nullptr);
    auto* checkpoint3 = new MerkleCheckpointNode(checkpoint2, "Optimize->C++");
    checkpoint3->addCoveredNode(emitNode);

    // Each checkpoint links to previous, forming merkle tree
    assert(checkpoint1->in(0) == nullptr);  // Root has no parent
    assert(checkpoint2->in(0) == checkpoint1);
    assert(checkpoint3->in(0) == checkpoint2);

    // Hashes form chain
    const auto& hash1 = checkpoint1->hash();
    const auto& hash2 = checkpoint2->hash();
    const auto& hash3 = checkpoint3->hash();

    // Different stages produce different hashes
    assert(hash1 != hash2);
    assert(hash2 != hash3);

    std::cout << "✓ Merkle tree chains compilation stages" << std::endl;

    // Cleanup
    delete checkpoint3;
    delete checkpoint2;
    delete checkpoint1;
    delete emitNode;
    delete optNode->in(1);
    delete optNode;
    delete srcNode;
}

int main() {
    std::cout << "Running Attestation & Anti-Cheat Node Tests" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;

    test_merkle_checkpoint();
    test_hash_witness();
    test_signature_point();
    test_tamper_guard();
    test_injection_barrier();
    test_determinism_fence();
    test_attestation_integration();

    std::cout << std::endl << "All attestation tests passed! ✓" << std::endl;
    return 0;
}
