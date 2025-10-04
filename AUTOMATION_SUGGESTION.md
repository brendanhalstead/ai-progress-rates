# Implementation Plan: Fast “Optimized Coding Labor” for ODE Simulations

> **Goal:** Replace the current *simple CES of (human labor `H`, inference compute `C`)* with a **behaviorally optimized CES** that endogenously allocates human time and automation compute across a continuum of task difficulties while keeping **per-timestep cost \~O(1)** (table lookups + a few ops), suitable for tight ODE loops.

---

## 0) Conceptual Overview (self-contained)

We model tasks on a **continuous index** $i\in[0,1]$ with **equal weight**. Harder tasks sit at larger $i$. Each task produces

$$
G(i)=h(i)+\eta(i)\,c(i),
$$

where $h(i)\ge 0$ is human input density and $c(i)\ge 0$ is automation-compute density. Totals at a timestep are fixed:

$$
\int_0^1 h(i)\,di=H,\qquad \int_0^1 c(i)\,di=C.
$$

Aggregate “coding labor” is CES:

$$
L=\Big(\int_0^1 G(i)^{\rho}\,di\Big)^{\!1/\rho},\quad \rho<1,\ \rho\neq 0.
$$

**Automation multiplier.** Each task has an **automation threshold** $E_{\text{aut}}(i)$ that is **nondecreasing** in $i$. Given current capability $E>0$,

$$
\eta(i)=
\begin{cases}
0,& E\le E_{\text{aut}}(i),\\[3pt]
\eta_{\text{init}}\Big(\dfrac{E}{E_{\text{aut}}(i)}\Big)^{\theta},& E>E_{\text{aut}}(i),
\end{cases}
$$

with fixed $\eta_{\text{init}}>0$ and slope $\theta>0$.

**Optimal policy structure (key fact):** With equal weights and monotone $\eta(i)$, the optimal allocation is **single-frontier**:

* There exists a cut $i_c\in[0,1]$:

  * $i<i_c$: **compute-only** region $(h=0,\ c>0)$
  * $i>i_c$: **human-only** region $(h>0,\ c=0)$
  * At the boundary (measure-zero), inputs may mix.

This enables a **closed-form reduction** so we can precompute a few one-dimensional arrays once, and then evaluate $L$ in **O(1)** time per ODE step.

---

## 1) Math That Powers the Fast Path (what we precompute)

Define

$$
\alpha=\frac{\rho}{1-\rho},\quad
\beta=\alpha\,\theta=\frac{\rho\,\theta}{1-\rho},\quad
\gamma=\frac{\theta}{1-\rho},\quad
\kappa=\frac{C}{H},\quad
S=\kappa\,\eta_{\text{init}}\,E^{\theta}.
$$

Let $B(x)=\displaystyle\int_{0}^{x} E_{\text{aut}}(u)^{-\beta}\,du$ (cumulative integral; depends **only** on $E_{\text{aut}}$).
Define three **lookup functions** (arrays on a grid):

$$
\begin{aligned}
F(x) &= \frac{B(x)\,E_{\text{aut}}(x)^{\gamma}}{1-x},\\[4pt]
Q(x) &= (1-x)^{1-\rho},\\[4pt]
R(x) &= (1-x)^{1-\rho}\,E_{\text{aut}}(x)^{-\theta}.
\end{aligned}
$$

Interpretation:

* $F$ links the frontier $x=i_c$ to $S$ via the **interior optimality condition** $F(i_c)=S$.
* $Q, R$ let us evaluate $L$ cheaply at the cut.

We also need $i_E=\sup\{x:\ E_{\text{aut}}(x)\le E\}$ (largest index whose threshold is met at capability $E$). The compute region cannot exceed $i_E$.

**Two cases at runtime**

1. **Interior (frontier feasible):** if $S \le F(i_E)$, then the cut solves $F(i_c)=S$ and

   $$
   L^\rho/H^\rho = Q(i_c) + S\,R(i_c).
   $$
2. **Boundary-limited (not enough tasks above threshold):** if $S > F(i_E)$, the cut saturates at $i_c=i_E$ and

   $$
   L^\rho/H^\rho = Q(i_E) + S^{\rho}\,B(i_E)^{\,1-\rho}.
   $$

   (This is the general expression that doesn’t assume the interior equality.)

Thus, **with $F,Q,R,B$ precomputed**, per-step evaluation is: compute $S$, invert $F$ if needed, then one addition and a power.

---

## 2) Deliverables & File Layout

Create a small self-contained module (Python/NumPy suggested):

```
coding_labor/
  __init__.py
  frontier.py          # precompute & runtime evaluation
  tests_frontier.py    # unit tests & property tests
  bench_frontier.py    # simple microbenchmarks
  README.md            # short usage notes
```

---

## 3) Public API (what other code will call)

```python
from typing import NamedTuple, Optional
import numpy as np

class FrontierPrecomp(NamedTuple):
    grid_i: np.ndarray            # shape [M], monotone in [0, 1-eps]
    Eaut:   np.ndarray            # shape [M], nondecreasing
    B:      np.ndarray            # shape [M]
    F:      np.ndarray            # shape [M], strictly increasing (except flats)
    Q:      np.ndarray            # shape [M]
    R:      np.ndarray            # shape [M]
    rho:    float
    theta:  float
    eta_init: float
    eps_i1: float                 # 1 - grid_i[-1], for safety

def precompute_frontier(Eaut_grid: np.ndarray,
                        rho: float,
                        eta_init: float,
                        theta: float) -> FrontierPrecomp:
    """One-time precompute of B, F, Q, R on a fixed i-grid aligned with Eaut_grid.
    Requirements: Eaut_grid is nondecreasing; rho < 1 and rho != 0; eta_init>0; theta>0.
    """

def coding_labor(H: float, C: float, E: float,
                 pc: FrontierPrecomp,
                 return_details: bool = False):
    """Fast evaluation at a timestep.
    Returns L (coding labor). If return_details=True, also returns dict with
    i_cut, i_E, case ('interior' or 'boundary'), and normalized terms."""
```

Optional derivative API for implicit ODE solvers:

```python
def coding_labor_with_grads(H, C, E, pc):
    """Returns L, and partials dL/dE, dL/dH, dL/dC."""
```

---

## 4) Implementation Steps

### 4.1 Precomputation (`precompute_frontier`)

1. **Input grid.** Accept a monotone grid `grid_i` implicitly from `Eaut_grid.shape`; set `grid_i = np.linspace(0.0, 1.0 - 1e-6, M)` if not provided. `M≈2048–8192` is fine.
2. **Validate monotonicity.** `np.all(np.diff(Eaut_grid) >= 0)`.
3. **Constants.**

   ```python
   alpha  = rho / (1.0 - rho)
   beta   = alpha * theta
   gamma  = theta / (1.0 - rho)
   ```
4. **B(x).** Cumulative trapezoid of `Eaut_grid**(-beta)` over `grid_i`. Normalize by the grid spacing `di`.

   ```python
   w = Eaut_grid ** (-beta)
   B = np.cumsum(0.5 * (w[1:] + w[:-1])) * di
   B = np.concatenate([[0.0], B])
   ```
5. **Q, R, F.**

   ```python
   one_minus_i = 1.0 - grid_i
   Q = one_minus_i ** (1.0 - rho)
   R = Q * (Eaut_grid ** (-theta))
   F = (B * (Eaut_grid ** gamma)) / one_minus_i
   ```
6. **Monotonic fixups.** Enforce `F` strictly increasing by `np.maximum.accumulate(F + tiny)` if needed.
7. **Return `FrontierPrecomp`.**

**Complexity:** O(M). Happens once at model init or parameter change.

---

### 4.2 Runtime Evaluation (`coding_labor`)

**Inputs:** `H, C, E, pc` (precomp).
**Derived:** `kappa = C / H`, `S = kappa * pc.eta_init * E**pc.theta`.

1. **Find capability-limited index $i_E$.**

   * Binary search on `pc.Eaut` to locate the **rightmost** index with `Eaut<=E`.
   * Linearly interpolate to a fractional index `i_E`.
2. **Case split** using `pc.F`:

   * Interpolate `F(i_E)` via the same fractional index.
   * If `S <= F(i_E)`: **interior**

     * Invert `F`: find `i_c` s.t. `F(i_c)=S` (bisection on the grid + linear interpolation).
     * Compute `human = Q(i_c)` and `comp = S * R(i_c)` (both by interpolation).
     * `L_norm_rho = human + comp`.
   * Else: **boundary-limited**

     * Set `i_c = i_E`.
     * `human = Q(i_E)`.
     * `comp  = S**pc.rho * B(i_E)**(1.0 - pc.rho)`.
     * `L_norm_rho = human + comp`.
3. **Scale back by H (homogeneity).**

   * `L = H * (L_norm_rho ** (1.0 / pc.rho))`.
4. **Return.** If `return_details=True`, also return diagnostic dict.

**Per-step cost:** \~two binary searches + a handful of flops; can be reduced to **O(1)** with inverse LUTs (see §7).

---

## 5) Optional: Gradients for ODE Jacobians

If needed by stiff/implicit solvers:

* $dS/dE = \theta\,S/E$, $dS/dH = -S/H$, $dS/dC = S/C$.
* **Interior case** $S\le F(i_E)$: $F(i_c)=S\Rightarrow di_c/dS = 1/F'(i_c)$.

  * Precompute a numerical `Fprime` by centered differences; interpolate at `i_c`.
  * With $V(S)=Q(i_c)+S R(i_c)$,

    $$
    \frac{dV}{dS}=Q'(i_c)\frac{1}{F'(i_c)}+R(i_c)+S\,R'(i_c)\frac{1}{F'(i_c)}.
    $$
  * Chain with $dS/dE, dS/dH, dS/dC$; finally

    $$
    \frac{\partial L}{\partial (\cdot)} = H\cdot\frac{1}{\rho}\,V^{\frac{1}{\rho}-1}\,\frac{dV}{d(\cdot)}\ +\ \mathbb{1}_{(\cdot)=H}\cdot V^{\frac{1}{\rho}}.
    $$
* **Boundary case**: use $V=Q(i_E)+S^{\rho}B(i_E)^{1-\rho}$; treat $i_E$ via the inverse of `Eaut` (numerical derivative) if you want exactness. Often you can **freeze $i_E$** within a step for stability and skip its tiny derivative.

(If gradients aren’t essential, omit this section from the first implementation.)

---

## 6) Integration Guide (drop-in replacement)

1. **Config / Params**

   * Provide $\rho<1$ (avoid $\rho \to 1^-$), $\eta_{\text{init}}>0$, $\theta>0$.
   * Provide a **monotone** `Eaut_grid[i]` on `i∈[0,1]` (e.g., 2–8k points).
2. **Init stage (once):**

   ```python
   pc = precompute_frontier(Eaut_grid, rho, eta_init, theta)
   ```
3. **ODE right-hand-side (each step):**

   ```python
   L = coding_labor(H_t, C_t, E_t, pc)  # replace old simple-CES call
   # use L in your ODEs that update E, etc.
   ```
4. **Feature flag / fallback:**

   * Keep a runtime switch:

     * `"coding_labor.mode" ∈ {"simple_ces", "optimal_ces"}`
   * Old simple CES (current behavior) for sanity checks:

     $$
     L_{\text{simple}} = \left(a\,H^{\rho} + (1-a)\,C^{\rho}\right)^{1/\rho}
     $$

     with an app-specific `a∈(0,1)`.

---

## 7) Performance Tips

* **Grid size:** `M=4096` is typically enough. Increase if `Eaut` is very steep.
* **Inverse LUT (optional O(1)):** Offline, tabulate $i = F^{-1}(s)$ for `s` on a **log-spaced** range that covers all plausible `S`. At runtime, do a single interpolation instead of a binary search.
* **Stability:** Avoid `i=1` exactly (use `1-1e-6`). Clamp `S` to `[F(grid[0]), F(grid[i_E])]` before inversion.
* **Units / scaling:** If `E` spans many decades, compute `S` in log-space to avoid overflow: `logS = logkappa + log(eta_init) + theta*logE`.

---

## 8) Testing Plan (must pass before shipping)

1. **Shape tests**

   * Monotonicity: `L(H,C,E)` increases in `H` and in `E` (holding the other fixed and `C/H` fixed).
   * If `E < Eaut_grid[0]`: `L ≈ H` (no tasks automatable).
   * If `C=0`: `L = H`.
2. **Continuity**

   * Sweep `E` across a wide range; ensure no NaNs, no jumps.
3. **Interior vs Boundary parity**

   * Construct cases with both `S ≤ F(i_E)` and `S > F(i_E)`; assert `L` is continuous where they meet.
4. **Discrete sanity check**

   * Create a small **discrete** benchmark (N=200 tasks). Solve the discrete KKT optimum (by sort + two bisections) and compare to the continuum implementation evaluated on the same `Eaut` curve. Target relative error `< 1e-3`.
5. **Speed**

   * Benchmark 1e5 calls on a laptop; target **< 1–2 µs per call** (NumPy).

---

## 9) Edge Cases & Guards

* Enforce `rho < 1` and `rho != 0`. (Cobb-Douglas limit can be added later.)
* If `E` or `C/H` are outside the LUT range, **clamp** and log a warning once.
* If `Eaut` has flats, `F` may have flats; the `max.accumulate` trick ensures monotone invertibility.
* If `Eaut[0] == 0`, add a tiny floor `Eaut = np.maximum(Eaut, 1e-12)`.

---

## 10) Minimal Pseudocode

```python
def precompute_frontier(Eaut, rho, eta_init, theta):
    M = len(Eaut)
    i = np.linspace(0.0, 1.0 - 1e-6, M)
    di = i[1] - i[0]

    alpha  = rho / (1.0 - rho)
    beta   = alpha * theta
    gamma  = theta / (1.0 - rho)

    w = Eaut ** (-beta)
    B = np.concatenate([[0.0], np.cumsum(0.5*(w[1:]+w[:-1])) * di])

    one_minus_i = 1.0 - i
    Q = one_minus_i ** (1.0 - rho)
    R = Q * (Eaut ** (-theta))
    F = (B * (Eaut ** gamma)) / one_minus_i
    F = np.maximum.accumulate(F)  # enforce monotone

    return FrontierPrecomp(i, Eaut, B, F, Q, R, rho, theta, eta_init, 1.0-i[-1])

def interp(x_grid, y_grid, x):
    # linear interpolation helper; assumes x_grid monotone
    j = np.searchsorted(x_grid, x, side='right') - 1
    j = np.clip(j, 0, len(x_grid)-2)
    t = (x - x_grid[j]) / (x_grid[j+1]-x_grid[j])
    return (1-t)*y_grid[j] + t*y_grid[j+1]

def invert_monotone(x_grid, y_grid, y):
    # find x with y_grid(x)=y; binary search + local linear interpolate
    lo, hi = 0, len(x_grid)-1
    if y <= y_grid[lo]: return x_grid[lo]
    if y >= y_grid[hi]: return x_grid[hi]
    while hi - lo > 1:
        mid = (hi + lo)//2
        if y_grid[mid] < y: lo = mid
        else: hi = mid
    # linear segment [lo,hi]
    y0, y1 = y_grid[lo], y_grid[hi]
    t = (y - y0) / (y1 - y0 + 1e-18)
    return (1-t)*x_grid[lo] + t*x_grid[hi]

def coding_labor(H, C, E, pc, return_details=False):
    kappa = C / max(H, 1e-18)
    S = kappa * pc.eta_init * (E ** pc.theta)

    # capability-limited index i_E (invert Eaut <= E)
    j = np.searchsorted(pc.Eaut, E, side='right') - 1
    j = np.clip(j, 0, len(pc.grid_i)-2)
    # fractional i_E using the local segment
    i0, i1 = pc.grid_i[j], pc.grid_i[j+1]
    e0, e1 = pc.Eaut[j], pc.Eaut[j+1]
    tE = 0.0 if e1==e0 else np.clip((E - e0) / (e1 - e0), 0.0, 1.0)
    i_E = (1-tE)*i0 + tE*i1

    F_iE = interp(pc.grid_i, pc.F, i_E)

    if S <= F_iE:
        # interior
        i_c = invert_monotone(pc.grid_i, pc.F, S)
        human = interp(pc.grid_i, pc.Q, i_c)
        comp  = S * interp(pc.grid_i, pc.R, i_c)
        L_norm_rho = human + comp
        case = "interior"
    else:
        # boundary-limited
        i_c = i_E
        human = interp(pc.grid_i, pc.Q, i_c)
        B_i  = interp(pc.grid_i, pc.B, i_c)
        comp = (S ** pc.rho) * (B_i ** (1.0 - pc.rho))
        L_norm_rho = human + comp
        case = "boundary"

    L = H * (L_norm_rho ** (1.0 / pc.rho))

    if return_details:
        return L, {"i_cut": i_c, "i_E": i_E, "case": case,
                   "human_term": human, "comp_term": comp}
    return L
```

---

## 11) Migration Checklist

* [ ] Add new module; wire config flag `coding_labor.mode`.
* [ ] Load `Eaut_grid` from model config; validate monotonicity.
* [ ] Call `precompute_frontier` during model initialization.
* [ ] Replace old CES call in the ODE RHS with `coding_labor`.
* [ ] Run unit tests + discrete KKT parity test.
* [ ] Run scenario benchmarks to verify total wall-time budgets.
* [ ] Compare trajectories under both modes on a few seeds; sanity-check qualitative behavior.

---

## 12) Notes & Limitations

* Keep $\rho < 1$ and $\rho \neq 0$. (Cobb-Douglas $\rho\to 0$ can be added with log-limits later.)
* This implementation assumes equal task weights and a single monotone frontier. If you later add heterogeneity in weights or non-monotone $E_{\text{aut}}$, the single-cut property may fail—different algorithm required.
* If `E` wanders outside your expected range, consider a **log-spaced inverse LUT** on $S$ to keep per-step cost constant without binary searches.

---

**Done.** This plan gives you a drop-in, near-constant-time `coding_labor` for ODE simulations with a one-time precompute and clear numerical behavior.
