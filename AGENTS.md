# CLAUDE.md

This file provides guidance to   Code when working with code in this repository.

## Project: Trustless cpp2 Compiler with Betanet Transport

Self-hosting cpp2 transpiler distributing content-addressable modules over Betanet's censorship-resistant transport, integrated with spartan_agent fractal gap analysis for compilation error resolution.

## Architecture

Three-layer system:
- **cpp2 Compiler (cppfront)**: Transpiles cpp2 → C++
- **Spartan Agent Gap Analysis**: Fractal gap analysis for compilation errors and network failures
- **Betanet Transport (HTX/HTXQUIC)**: Censorship-resistant module distribution via mixnodes

### Data Flow
```
cpp2 source → cppfront → import<${hash}.cppm> → betanet://resolve → mixnode fetch
                ↓                                      ↓                    ↓
           compilation errors              service discovery        cached locally
                ↓                                      ↓                    ↓
           spartan_agent                      HTX transport          load module
           gap closure                        verification
```

### Module Import Pattern
```cpp
import<betanet://a1b2c3d4e5f6789abcdef...>.cppm;  // Service pubkey hash
```

Service addresses: `betanet://<SHA-256(service-pubkey)>/modules/${cas_hash}`

## Development Commands

### Compilation
```bash
# Transpile cpp2 to C++
cppfront source.cpp2

# Compile with debug symbols for reverse engineering validation
g++ -O0g -o debug_binary source.cpp

# Compile optimized for reverse engineering comparison
g++ -O2 -o optimized_binary source.cpp
```

### Testing: TDD Reverse Engineering Pipeline
Test cycle validates that compiled binaries can be reverse engineered back to semantically equivalent cpp2:
1. Transpile cpp2 → C++
2. Compile to both -O0g (debug) and -O2 (optimized) binaries
3. Reverse engineer both binaries
4. Verify semantic equivalence with original cpp2 source

This creates a feedback loop where compilation fixes improve reverse engineering accuracy.

## Core Components

### Spartan Agent (Python) - Located in /Users/jim/work/build-trap-agent/
- `gap_catalog.py`: Gap patterns including Betanet network errors (HTX timeout, service discovery failure, Cashu voucher issues)
- `meta_agent.py`: Enhanced with `BetanetMetaAgent` for mixnode-aware gap closure
- Network error classification: `classify_network_gap()` handles transport failures
- Terminology: Use "gap closure" and "fractal decomposition", not "remediation"

### Betanet Integration (C++)
- `BetanetCASResolver`: HTX/HTXQUIC client for module resolution with mixnode retry logic
- Import resolution: Handles `betanet://`, `https://` (dev fallback), and local cache
- CAS verification: Content-addressable storage with hash validation

### Network Error Types
- `betanet_transport_failure`: HTX connection failed → retry with different mixnode path
- `betanet_discovery_failure`: Service not found → check multi-chain finality status
- Mixnode path failures, service discovery delays, Cashu voucher depletion, HTX timeouts

## Betanet Protocol Requirements

- **Transport**: HTX/HTXQUIC with covert indistinguishability
- **Routing**: ≥2 mixnode hops until trust ≥0.8
- **Cryptography**: X25519-Kyber768 (mandatory from 2027-01-01)
- **Economic**: Optional Cashu vouchers (128-byte, Lightning settlement)
- **Governance**: 2-of-3 multi-chain finality (Handshake L1, Filecoin FVM, Ethereum L2)

## Implementation Phases

### Phase 1: Foundation
- Extend spartan_agent with Betanet gap patterns in `gap_catalog.py`
- Design `betanet://` address scheme for cpp2 services
- Simulate mixnode failures for gap analysis validation
- HTX dependency (waiting on $12K bounty completion)

### Phase 2: Transport Integration
- Implement `BetanetCASResolver` for module fetching over HTX
- Mixnode retry logic in spartan_agent
- Service registration with multi-chain governance
- Performance baseline: measure compilation overhead

### Phase 3: Production Hardening
- Validate compilation under adverse network conditions
- Optimize Cashu voucher usage
- X25519-Kyber768 transition preparation
- Submit bounty proposals to $43K USDC fund

## Bootstrap Strategy

```
Traditional Internet → Basic cppfront → Enhanced cppfront → Betanet-native
        ↓                    ↓                ↓                  ↓
Direct download      CAS-aware imports  HTX transport    Mixnode routing
No verification      Hash verification  Service discovery Anti-censorship
```

## Risk Mitigation

- **HTX delay**: Develop against mock transport layer (HIGH severity)
- **Mixnode latency**: Aggressive local caching, parallel path probing (MEDIUM)
- **Cashu costs**: Free node fallbacks, compiler service subsidization (LOW)