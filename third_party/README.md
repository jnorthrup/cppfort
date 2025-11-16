# Third Party (vendoring)

This directory is for vendoring small third-party components used by the repository in the build.

For CAS (BLAKE3) we recommend one of the following approaches:

1. Install system blake3 (macOS Homebrew: `brew install blake3`) then run CMake with `-DUSE_BLAKE3=ON`.
2. Vendor the BLAKE3 sources under `third_party/blake3/` and add a `CMakeLists.txt` that builds a small library.
   The build system will find the library automatically if installed or when the vendored target is added to stage0 targets.

Please ensure any vendored code includes appropriate LICENSING information and attribution.
