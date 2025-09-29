# cpp2: In-situ borrow checking & escape analysis (backdrop)

Reference: [`src/stage0/ast.h`](src/stage0/ast.h:1)

Summary

- Replace tracing GC by proving at compile-time which allocations are non-escaping or by enforcing lightweight borrow constraints; fallback to GC/RC when proof fails.

Target IR

- Operate on SSA IR (single-assignment, def-use, dominance) to compute precise flow-sensitive lifetimes.

Pipeline

1. Identify allocation sites in SSA (alloc/new).
2. Run escape analysis following SSA uses, memory stores, PHI/captures and interprocedural summaries.
3. Convert non-escaping sites to stack/in-situ storage and remove GC bookkeeping.
4. For uncertain/escaping sites, generate lifetime/borrow constraints and (a) prove via subsumption and remove GC, (b) emit thin runtime guards, or (c) use GC/RC.

Constraint model

- Lifetime vars: each allocation has $L_a$.
- Uses impose $L_{use} \sqsubseteq L_a$ or $L_a \sqsubseteq L_{scope}$.
- PHI/merge: $L_{phi} = \sqcup_i L_{in_i}$.
- Aliasing/permissions: track read vs mutable; enforce linearity or permission sets.

SSA advantages

- Def/use graphs and dominance give minimal live ranges; phi nodes expose merges for lifetime joins; value numbering shrinks aliases.

Soundness sketch

- Invariant: at any program point p, every live reference r refers to allocation a with $L_a \succeq L_p$, or safety is ensured by a runtime guard.
- Subsumption is monotone: proving $L_x \sqsubseteq L_y$ ensures x outlives y; this prevents dangling references.

Practical pitfalls

- Interprocedural calls require summaries or conservative escape; inline hot callees to gain precision.
- Closures and captures usually escape unless proven transient.
- Interior mutability, aliasing, and concurrency complicate static proof; fallback to runtime checks or GC.

Engineering recommendations

- Two-tier system: fast conservative escape analysis and a targeted subsumption solver for ambiguous hot paths.
- Cache parametric interprocedural lifetime summaries.
- Emit runtime guards for rare escape paths instead of keeping GC broadly.
- Preserve or recompute SSA and constraints across transformations.
- Keep GC/RC fallback; ensure compiler emits boxed representations when solver cannot prove safety.

Bottom line

- Modeling lifetimes and permissions as a subsumption lattice over SSA-derived constraints enables sound elimination of GC in many cases; use staged precision and fallbacks to balance compile cost vs runtime safety.