#pragma once
// Cryptographic attestation and anti-cheat system for n-way compiler
// Ensures deterministic compilation and detects tampering

#include <string>
#include <vector>
#include <memory>
#include <array>
#include <chrono>
#include <unordered_map>

namespace attestation {

// SHA3-512 hash (64 bytes)
using Hash = std::array<uint8_t, 64>;

// Ed25519 signature (64 bytes)
using Signature = std::array<uint8_t, 64>;

// Ed25519 public key (32 bytes)
using PublicKey = std::array<uint8_t, 32>;

// Ed25519 private key (64 bytes)
using PrivateKey = std::array<uint8_t, 64>;

// Merkle tree node
struct MerkleNode {
    Hash hash;
    std::shared_ptr<MerkleNode> left;
    std::shared_ptr<MerkleNode> right;
    size_t index;
    size_t level;
};

// Compilation attestation record
struct CompilationRecord {
    // Source information
    std::string source_file;
    Hash source_hash;
    uint64_t source_size;
    uint64_t source_timestamp;

    // Compilation information
    std::string compiler_version;
    std::string target_triple;
    std::vector<std::string> compiler_flags;
    uint64_t compilation_timestamp;

    // IR information
    Hash ir_hash;
    size_t ir_node_count;

    // Output information
    std::string output_file;
    Hash output_hash;
    uint64_t output_size;

    // Deterministic build ID
    Hash build_id;

    // Signature
    Signature signature;
    PublicKey signer_key;

    // Chain of trust
    std::vector<Hash> dependency_hashes;
    Hash parent_build;  // Previous compilation in chain
};

// Attestation manifest for multi-file projects
struct AttestationManifest {
    std::string project_name;
    std::string project_version;
    uint64_t manifest_timestamp;

    // Individual file records
    std::vector<CompilationRecord> compilations;

    // Merkle tree root for all compilations
    Hash merkle_root;

    // Project signature
    Signature project_signature;
    PublicKey project_key;

    // Build reproducibility info
    std::string build_environment;
    std::unordered_map<std::string, std::string> environment_vars;
};

// SHA3-512 hasher
class SHA3Hasher {
public:
    SHA3Hasher();
    ~SHA3Hasher();

    void update(const void* data, size_t len);
    void update(const std::string& str);
    Hash finalize();

    // Static convenience methods
    static Hash hash(const void* data, size_t len);
    static Hash hash(const std::string& str);
    static Hash hashFile(const std::string& filepath);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Ed25519 signature system
class Ed25519Signer {
public:
    Ed25519Signer();
    ~Ed25519Signer();

    // Key generation
    void generateKeypair(PublicKey& pub, PrivateKey& priv);

    // Signing and verification
    Signature sign(const Hash& hash, const PrivateKey& key);
    bool verify(const Hash& hash, const Signature& sig, const PublicKey& key);

    // Key serialization
    std::string serializePublicKey(const PublicKey& key);
    std::string serializePrivateKey(const PrivateKey& key);
    PublicKey deserializePublicKey(const std::string& str);
    PrivateKey deserializePrivateKey(const std::string& str);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Merkle tree builder
class MerkleTree {
public:
    MerkleTree();

    // Add leaves (compilation hashes)
    void addLeaf(const Hash& hash);
    void addLeaves(const std::vector<Hash>& hashes);

    // Build tree
    void build();

    // Get root hash
    Hash getRoot() const;

    // Generate proof for a leaf
    std::vector<Hash> getProof(size_t leaf_index) const;

    // Verify proof
    static bool verifyProof(const Hash& leaf, const std::vector<Hash>& proof,
                           const Hash& root);

    // Tree statistics
    size_t getHeight() const;
    size_t getLeafCount() const;

private:
    std::vector<Hash> leaves;
    std::shared_ptr<MerkleNode> root;
    std::vector<std::vector<std::shared_ptr<MerkleNode>>> levels;

    std::shared_ptr<MerkleNode> buildLevel(
        const std::vector<std::shared_ptr<MerkleNode>>& nodes);
    Hash combineHashes(const Hash& left, const Hash& right);
};

// Deterministic compilation controller
class DeterministicCompiler {
public:
    DeterministicCompiler();

    // Set compilation parameters for determinism
    void setSeed(uint64_t seed);
    void setTimestamp(uint64_t timestamp);
    void disableTimestamps();
    void disableRandomness();
    void sortSymbols();
    void normalizeFilePaths();

    // Compilation tracking
    void beginCompilation(const std::string& source_file);
    void recordIR(const void* ir_data, size_t size);
    void recordOutput(const std::string& output_file);
    CompilationRecord endCompilation();

    // Deterministic ID generation
    Hash generateBuildId(const CompilationRecord& record) const;

    // Environment normalization
    void normalizeEnvironment();
    std::unordered_map<std::string, std::string> captureEnvironment() const;

private:
    uint64_t deterministic_seed = 0;
    uint64_t deterministic_timestamp = 0;
    bool timestamps_disabled = false;
    bool randomness_disabled = false;
    bool sort_symbols = false;
    bool normalize_paths = false;

    CompilationRecord current_record;
    SHA3Hasher hasher;

    std::string normalizePath(const std::string& path) const;
};

// Anti-cheat detector
class AntiCheatDetector {
public:
    AntiCheatDetector();

    // Compiler binary verification
    bool verifySelf(const std::string& expected_hash);
    Hash computeSelfHash();

    // Code injection detection
    bool detectInjection();
    void registerHook(void* address, const std::string& name);
    bool verifyHooks();

    // AST tampering detection
    struct ASTDiff {
        enum DiffType { Added, Removed, Modified };
        DiffType type;
        std::string node_type;
        std::string location;
        std::string description;
    };

    std::vector<ASTDiff> compareAST(const void* ast1, const void* ast2);
    bool detectASTTampering(const void* ast, const Hash& expected_hash);

    // Runtime attestation
    void installRuntimeChecks();
    bool verifyRuntimeIntegrity();

    // Memory protection
    void protectMemoryRegion(void* addr, size_t size);
    bool verifyMemoryIntegrity(void* addr, size_t size, const Hash& expected);

private:
    struct HookInfo {
        void* address;
        std::string name;
        Hash original_hash;
    };

    std::vector<HookInfo> registered_hooks;
    std::unordered_map<void*, Hash> memory_hashes;

    bool checkPtraceStatus();
    bool checkDebuggerPresence();
    bool checkEnvironmentIntegrity();
};

// Attestation chain manager
class AttestationChain {
public:
    AttestationChain();

    // Chain operations
    void addRecord(const CompilationRecord& record);
    bool verifyChain() const;
    CompilationRecord getRecord(size_t index) const;
    size_t getChainLength() const;

    // Chain persistence
    void saveChain(const std::string& filepath);
    void loadChain(const std::string& filepath);

    // Chain validation
    bool validateRecord(const CompilationRecord& record) const;
    bool validateTransition(const CompilationRecord& prev,
                           const CompilationRecord& next) const;

    // Trust management
    void addTrustedKey(const PublicKey& key);
    void revokeTrustedKey(const PublicKey& key);
    bool isKeyTrusted(const PublicKey& key) const;

private:
    std::vector<CompilationRecord> chain;
    std::vector<PublicKey> trusted_keys;
    MerkleTree merkle_tree;

    bool verifySignature(const CompilationRecord& record) const;
    bool verifyHashes(const CompilationRecord& record) const;
    bool verifyDeterminism(const CompilationRecord& record) const;
};

// Binary signature embedder
class BinarySignatureEmbedder {
public:
    enum Format { ELF, PE, MachO };

    BinarySignatureEmbedder(Format format);

    // Embed signature in binary
    bool embedSignature(const std::string& binary_path,
                       const Signature& sig,
                       const CompilationRecord& record);

    // Extract signature from binary
    bool extractSignature(const std::string& binary_path,
                         Signature& sig,
                         CompilationRecord& record);

    // Verify binary signature
    bool verifyBinary(const std::string& binary_path,
                     const PublicKey& key);

private:
    Format format;

    // Format-specific embedding
    bool embedELF(const std::string& path, const void* data, size_t size);
    bool embedPE(const std::string& path, const void* data, size_t size);
    bool embedMachO(const std::string& path, const void* data, size_t size);

    // Format-specific extraction
    bool extractELF(const std::string& path, void* data, size_t& size);
    bool extractPE(const std::string& path, void* data, size_t& size);
    bool extractMachO(const std::string& path, void* data, size_t& size);
};

// Main attestation system
class AttestationSystem {
public:
    static AttestationSystem& getInstance();

    // Initialize system
    void initialize();

    // Compilation attestation
    void beginCompilation(const std::string& source);
    void attestIR(const void* ir, size_t size);
    void attestOutput(const std::string& output);
    CompilationRecord finalizeCompilation();

    // Verification
    bool verifyCompilation(const CompilationRecord& record);
    bool verifyBinary(const std::string& binary_path);

    // Anti-cheat
    bool performSelfCheck();
    bool detectTampering();

    // Chain management
    void addToChain(const CompilationRecord& record);
    bool verifyChain();

    // Key management
    void generateKeypair();
    PublicKey getPublicKey() const;
    void loadPrivateKey(const std::string& keyfile);
    void savePrivateKey(const std::string& keyfile);

private:
    AttestationSystem();
    ~AttestationSystem();

    DeterministicCompiler deterministic;
    AntiCheatDetector anticheat;
    AttestationChain chain;
    Ed25519Signer signer;
    SHA3Hasher hasher;

    PublicKey public_key;
    PrivateKey private_key;
    bool keys_loaded = false;

    CompilationRecord current_compilation;
};

} // namespace attestation