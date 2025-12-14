Imported tests from https://github.com/hsutter/cppfront (subtree clone)

Details:
- Source: `third_party/cppfront` repository
- Path: `third_party/cppfront/regression-tests`
- Upstream commit: 1a6062a

What is included:
- All `.cpp2` regression test inputs copied into `corpus/inputs/`.
- `corpus/sha256_database.txt` contains SHA256 hashes for all input `.cpp2` files.

Notes:
- The `[outputs]` section in `corpus/sha256_database.txt` is intentionally empty; to populate it, run `cppfront` on the inputs and record output checksums.
- The clone is currently committed as an embedded repository at `third_party/cppfront`. Consider converting to a `git submodule` if you want downstream clones to fetch it automatically.
