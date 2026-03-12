# cppfort

`cppfort` is currently being restarted through `selfhost/`.

The active direction is to bring TrikeShed-style abstractions over as a `cpp2` port, because that is the simpler path to a coherent compiler restart than trying to clean everything up first at the LLVM compilation layer. MLIR, Sea-of-Nodes, and the earlier `cppfront` corpus conveyor still matter as downstream architectural intent and project history, but they are not the current top-level operating contract.

## Current contract

- Use `cmake` and `ninja` as the supported build tools.
- Treat `selfhost/` as the live restart path and current source of truth.
- Use `selfhost_bootstrap_smoke` as the authoritative top-level build target.
- Treat `old/cppfort` only as a temporary bootstrap bridge used by the current smoke target.
- Treat the conveyor/`cppfront` corpus workflow as historical or downstream work, not the main entrypoint today.

## What `manifold` means here

In this repo, `manifold` means compiler-process guidance:

- charts, atlases, and coordinates for describing program forms
- transitions that guide normalization and lowering
- a way to organize movement between representations without losing intent

It does **not** mean:

- model training
- learned classification
- embeddings
- statistical inference

## Build

The verified build path today is:

```bash
cmake -S . -B build -G Ninja
ninja -C build selfhost_bootstrap_smoke
```

That path exercises the active self-host restart flow. At the moment, the smoke target still reaches through `old/cppfort` as a bootstrap bridge, but that bridge is transitional rather than authoritative.

## Repo layout

- `selfhost/`: the live restart path and the directory to treat as current work
- `old/`: archived and bootstrap-compatibility material; useful for bridging, not source truth
- `old/cppfort`: temporary bootstrap support currently used by `selfhost_bootstrap_smoke`
- `src/`: retired in the current worktree and not the active implementation path

## Architectural direction

The near-term goal is to recover a clean self-hosting path around the `selfhost/` restart and the `cpp2`/TrikeShed abstraction port. Once that path is solid, the longer arc still points toward richer normalization and lowering work, including MLIR and Sea-of-Nodes style back-end structure.

The earlier conveyor flow built around the `cppfront` corpus remains useful as downstream validation, historical context, and potential future integration surface. It is no longer the first workflow to understand or the first cleanup target to optimize.

## References

- [cppfront](https://github.com/hsutter/cppfront)
- [MLIR](https://mlir.llvm.org/)
- [Sea of Nodes IR](https://github.com/SeaOfNodes/Simple)
