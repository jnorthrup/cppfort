# cppfort

This repository implements a graph-first transpilation pipeline for C/C++/CPP2 using
orbit-based scanning and reversible graph primitives. See `ARCHITECTURE.md` and
`CODING_STANDARDS.md` for design, coding, and architectural details.

Quick notes:

- We encourage the use of `clang-format` configured at the repository root.
 - A minimal `cpp2_cas` helper is available under `src/stage0/` for rewriting
	 fenced cpp2 code blocks (```cpp2 ... ```) into CAS placeholders. This is a
	 placeholder; a BLAKE-based solution is planned.
