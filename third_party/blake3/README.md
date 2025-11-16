# BLAKE3 (vendored)

This directory optionally contains a vendored copy of the BLAKE3 C implementation used by
`src/stage0/cpp2_cas` when `VENDOR_BLAKE3` is enabled.

The project uses CMake `FetchContent` to download a stable release of BLAKE3 into the build
directory when asked, and exposes a small static library target (`blake3_vendor`) that
other targets (e.g., `stage0_lib`) can link against.

If you prefer to use a system-provided `blake3` installation, set `VENDOR_BLAKE3=OFF` when
running CMake; the build will attempt to find a system `libblake3` first.

License & Attribution
---------------------
The BLAKE3 implementation is Copyright (c) The BLAKE3 Authors and licensed under the MIT
license (see the upstream repository). When vendoring BLAKE3 into this repository, ensure
the upstream license is included and follow its attribution requirements.

Usage (CMake)
--------------
By default `VENDOR_BLAKE3` is ON. To switch it off:

```bash
cmake -S . -B build -DVENDOR_BLAKE3=OFF
cmake --build build
```

If `VENDOR_BLAKE3` is ON, the top-level `CMakeLists.txt` adds this directory and the vendored
BLAKE3 library will be used for CAS computation.
