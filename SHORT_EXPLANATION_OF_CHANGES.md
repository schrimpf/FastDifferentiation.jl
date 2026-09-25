# Summary of Changes: D* Derivative Graph Factorization for Multi-Output Functions

This pull request resolves long-standing issues in the $D^*$ derivative graph factorization algorithm for functions $f: \mathbb{R}^n \to \mathbb{R}^m$ with densely interconnected expression DAGs (specifically GitHub issues [#23](https://github.com/brianguenter/FastDifferentiation.jl/issues/23), [#65](https://github.com/brianguenter/FastDifferentiation.jl/issues/65), and [#108](https://github.com/brianguenter/FastDifferentiation.jl/issues/108)).

---

## 1. Algorithmic Context & Problem Formulation

In $D^*$, the derivative of a multi-output function $f: \mathbb{R}^n \to \mathbb{R}^m$ is represented by a directed acyclic derivative graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Each directed edge $e = (u, v)$ represents the local partial derivative $\frac{\partial v_u}{\partial v_v}$. Each of the $n \times m$ constituent functions
$$f_{ij} = \frac{\partial f_i}{\partial x_j}$$
is evaluated as the **sum of path products** over all directed paths from root/output node $i$ ($1 \le i \le m$) to leaf/variable node $j$ ($1 \le j \le n$):
$$f_{ij} = \sum_{\pi \in \Pi(i, j)} \prod_{e \in \pi} \text{value}(e)$$

Factorization reduces the number of multiplications and additions by identifying **factor subgraphs** $[b, c]$:
- In a **dominator subgraph** ($b \text{ dom } c$), factor node $b$ dominates factor base $c$ (every path from $c$ to root $i$ passes through $b$).
- In a **postdominator subgraph** ($b \text{ pdom } c$), factor node $b$ postdominates factor base $c$ (every path from $c$ to variable leaf $j$ passes through $b$).

Evaluating $[b, c]$ yields an equivalent factored subgraph edge $S$ from $b$ to $c$. In $f: \mathbb{R}^1 \to \mathbb{R}^1$, all paths in $[b, c]$ belong to the single derivative function. However, in $f: \mathbb{R}^n \to \mathbb{R}^m$ (Section 4.1 of the paper), edges inside $[b, c]$ are shared across distinct constituent functions $f_{ij}$.

---

## 2. Root Causes of Failures in `main`

Prior to this branch, factorization on expressions with shared multi-output/multi-variable structure suffered from four interrelated failure modes:

1. **Path Reachability Mismatch & Erasure (Issue #23)**:
   In `old_edge_path`, candidate forward edges were filtered using `subset(reachable_mask, reachable_variables(x))` (or roots), where `reachable_mask` was the union of *all* reachable variables across the entire subgraph. If a path reached only a single variable $x_j$, `subset` evaluated to `false`, causing path discovery to fail and dropping valid derivative paths to zero (`0.0`).
2. **Lossy Reachability Collapsing (Issues #65 & #108)**:
   When evaluating factor subgraph $[b, c]$, `evaluate_subgraph` summed all paths into a single scalar expression and created a single replacement edge $S$ with the union mask. When paths within $[b, c]$ reached different subsets of variables or roots, this cross-contaminated reachability, assigning derivative path products to wrong $f_{ij}$ components and causing subsequent downstream factorizations to assert:
   `AssertionError: Should only be one path from root R to variable V`.
3. **Flawed Branching Fallback**:
   `is_branching` incorrectly flagged subgraphs as "branching" whenever two paths crossed a shared trunk edge, even when those paths belonged to disjoint variables. It routed evaluation to `evaluate_branching_subgraph`, which discarded reachability masks entirely.
4. **Path Traversal Crossover & Non-Dominance Splitting**:
   `next_valid_edge` did not enforce reachability mask continuity along traversed paths. Additionally, walking interior paths via `edge_path` during non-dominance splitting failed when internal branching existed.

---

## 3. Summary of Key Fixes

- **Reachability Continuity in Path Traversal** ([`src/FactorableSubgraph.jl`](src/FactorableSubgraph.jl)):
  `next_valid_edge`, `isa_connected_path`, and `edges_on_path` now accept and preserve a `path_mask`. Candidate edges are validated with `overlap(path_mask, non_dominance_mask(a, edge))`. `PathIterator` stores an independent copy of the starting edge's reachability mask.
- **Topological Edge Mask Initialization** ([`src/DerivativeGraph.jl`](src/DerivativeGraph.jl)):
  Added `initialize_edge_masks!` to rigorously propagate variable reachability upward from leaves and root reachability downward from roots, ensuring consistent initial bitmasks for all derivative edges.
- **Pruning Dead Ends in Subgraph Discovery** ([`src/FactorableSubgraph.jl`](src/FactorableSubgraph.jl)):
  `subgraph_edges` now executes a bidirectional search (forward BFS from $c$, backward BFS from $b$) to retain only edges that are both reachable from $c$ and can reach $b$.
- **Reachability-Partitioned Subgraph Factoring** ([`src/Factoring.jl`](src/Factoring.jl)):
  `evaluate_subgraph` groups paths through $[b, c]$ by their exact reachability signatures using dynamic programming (`_evaluate_subgraph_paths`). It replaces $[b, c]$ with partitioned replacement edges having independent masks, preserving distinct $f_{ij}$ reachabilities.
- **Boundary-Cut Non-Dominance Edge Splitting & Reset** ([`src/FactorableSubgraph.jl`](src/FactorableSubgraph.jl)):
  In `add_non_dom_edges!` and `reset_edge_masks!`, operations are performed directly on the boundary edges incident to `dominated_node(subgraph)` (`forward_edges(subgraph, dominated_node(subgraph))`). This cleanly eliminates the fragile internal path-walking (`edge_path`) while leaving interior bypass paths entering from outside $[b, c]$ intact.
- **Robust Path Evaluation for Factored Graphs** ([`src/Factoring.jl`](src/Factoring.jl)):
  `follow_path` employs memoized dynamic programming from root $i$ to variable $j$. In completely factored graphs, it follows the single path in $O(\text{depth})$; if residual branches remain, it correctly sums all valid path products without triggering assertions.
- **Idempotent DerivativeGraph Construction** ([`src/DerivativeGraph.jl`](src/DerivativeGraph.jl)):
  `create_NoOp` now unwraps pre-existing `NoOp` wrappers (`is_NoOp(root) ? children(root)[1] : root`), ensuring graph rebuilding preserves exact postorder node numbering.
- **Multi-Output Hessian Construction** ([`src/Jacobian.jl`](src/Jacobian.jl)):
  Updated `hessian` to differentiate the gradient vector through `jacobian(vec(gradient), variable_order)`, enabling full subexpression reuse across the entire Hessian.

---

## 4. Test Suite & Verification

All existing and newly added test suites pass with **0 failures and 0 errors**:
- **Existing Suite** ([`test/runtests.jl`](test/runtests.jl)): Fixed test typos and corrected analytical derivative reference in `jacobian` test item ($\frac{\partial}{\partial y}(x^2 y^3) = 3 x^2 y^2$ instead of $4 x^2 y^2$).
- **Complex Expressions Regression Suite** ([`test/test-complex-expressions.jl`](test/test-complex-expressions.jl)): Added rigorous numerical validation for issues [#23](https://github.com/brianguenter/FastDifferentiation.jl/issues/23), [#65](https://github.com/brianguenter/FastDifferentiation.jl/issues/65), and [#108](https://github.com/brianguenter/FastDifferentiation.jl/issues/108) against finite differences.
- **DifferentiationInterface Standard Scenarios** ([`test/test-differentiation-interface.jl`](test/test-differentiation-interface.jl)): All 170 scenarios of `DifferentiationInterfaceTest.default_scenarios()` pass (7,278 assertions).
- **OptimizationProblems Test Suite** ([`test/test-optimization-problems.jl`](test/test-optimization-problems.jl)): 52 unconstrained and constrained optimization problems from `OptimizationProblems.ADNLPProblems` pass 100%. (On `main`, this suite fails on `:helical` with a multi-path assertion error).
