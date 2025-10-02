# APEX Features — Swimlanes

## stage0

- Fix remaining emitter issues affecting regression suite
- Implement type inference for auto declarations
- Add lifetime analysis for Cpp2 contract semantics
- Implement capture analysis for lambdas

## ir

- Implement real Sea of Nodes IR replacing mock implementation
- Complete Band-based Sea of Nodes IR implementation

## stage1

- Connect stages to use IR for inter-stage communication
- Add module system for Cpp2 imports

## patterns

- Implement n-way lowering patterns for IR transformations

## optimization

- Implement GCM (Global Code Motion) optimization pass
- Implement CSE (Common Subexpression Elimination) pass
- Add constant folding optimization
- Add dead code elimination pass

## mlir_bridge

- Add direct MLIR anchoring for IR nodes

## testing

- Add comprehensive unit test suite
- Add triple induction validation framework

## ci

- Implement CI/CD pipeline with automated regression testing

## stage2

- Implement Stage2 anticheat attestation integration

## parser

- Implement BNFC grammar-based parser
- Add error recovery in parser for better diagnostics

## backend

- Add support for multiple backend targets

## build

- Implement incremental compilation support

## betanet

- Implement Betanet content-addressable module resolution
- Add HTX/HTXQUIC transport layer
- Implement mixnode routing for module distribution
- Add Cashu voucher integration for bandwidth payment
- Implement multi-chain finality verification

## docs

- Add documentation generation from source comments

