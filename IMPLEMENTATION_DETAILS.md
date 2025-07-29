# Implementation Details Supplement

This document enumerates the *engineering–level* behaviours of `progress_model.py` that are intentionally glossed over in `MODEL_EXPLANATION.md`.  It is aimed at developers who need to modify or extend the codebase and therefore should be aware of the extra robustness layers, heuristics and fall-backs baked into the implementation.

## 1. Optimisation & Parameter Estimation

* Primary optimiser: **L-BFGS-B** (bounded, quasi-Newton).
* Automatic fall-backs: if L-BFGS-B fails to converge, the code retries with **TNC** and then **SLSQP**.
* **Strategic starting points**: before each optimisation run the script fabricates a diverse set of initial parameter vectors (extremes, Latin-hyper-cube samples, constraint-informed guesses, small perturbations) to improve robustness.
* **Regularisation**: a **quartic** penalty is added for extreme elasticities, and a **quadratic** penalty is added for `alpha`/`software_progress_share` values that cling too tightly to boundaries; this discourages numerically fragile parameter regimes.
* **Constraint pre-screening**: each anchor constraint is checked for physical plausibility and evaluability with the initial parameter guess.  Infeasible constraints are excluded to prevent the optimiser from deadlocking.

## 2. Numerical Safeguards in Core Functions

### CES wrappers (`_ces_function`, `compute_cognitive_output`)
* Extensive **edge-case handling** for:
  * `rho \approx 0` (Cobb-Douglas limit)
  * `rho = 1` (perfect substitutes)
  * `rho \ll 0` (near-Leontief)
  * zero or negative inputs
* Over/under-flow protection using logs and conditional fall-backs to linear or min operations.
* All outputs are clamped to be **finite and non-negative**, with upper caps (e.g. `1e3`) to avoid explosive growth in downstream ODEs.

### Sigmoid for Automation Fraction
* Exact formula: `L / (1 + exp(−k·(P − P_mid)))` where:
  * `L`  = `automation_fraction_at_superhuman_coder`
  * `P_mid` = `progress_at_half_sc_automation`
  * `k`  = `automation_slope`
* Additional guards:
  * If exponent > 100 the function short-circuits to 0 (avoids overflow)
  * If exponent < −100 it short-circuits to `L` (underflow)
  * Fallback linear interpolation if `exp` raises an exception.

## 3. Integration Strategy (`integrate_progress`)

1. Attempts a sequence of SciPy integrators with progressively looser tolerances:
   * `RK45` (strict → relaxed)
   * `RK23`
   * `DOP853`
   * `Radau` (stiff)
2. Each RHS evaluation clamps state & derivative values into safe ranges.
3. If **all** SciPy solvers fail, a custom **Euler fallback** with adaptive step count (≥100) is executed, again with per-step clamping.
4. **ODE Step Size Logging**: The integration process logs detailed step size statistics including:
   * Minimum, maximum, and mean step sizes
   * Total number of integration steps
   * Warnings for very small step sizes (potential stiffness indicators)
   * Warnings for large step size variations (potential instability indicators)
   * Euler fallback step size information when scipy integrators fail
   * Configurable via `ODE_STEP_SIZE_LOGGING`, `ODE_SMALL_STEP_THRESHOLD`, and `ODE_STEP_VARIATION_THRESHOLD` in `model_config.py`

## 4. Normalisation Mechanics

* `progress_rate_normalization` is **re-computed automatically** at the start of every call to `estimate_parameters` so that the initial overall progress rate equals **1.0**.  This parameter is *always treated as fixed* during optimisation.
* `cognitive_output_normalization` is bounded to \[1e-5 , 0.1\] to prevent runaway scaling.

## 5. Additional Defensive Coding Patterns

* Every public helper validates **finiteness** and **sign** of its inputs.
* Warnings are logged for any clamping or unusual behaviour, facilitating downstream debugging without halting execution.
* The simulation state variables (`P`, `RS`) and their derivatives are hard-clipped to reasonable ranges to thwart numerical blow-ups.

## 6. Gaps Between Doc & Code

`MODEL_EXPLANATION.md` intentionally omits the safeguards above for clarity.  There is no functional mismatch—just extra robustness:

* The high-level formulas match 1:1 with the code.
* The model adds the normalized software progress rate (`S`) directly to the raw training compute input (`T`). The resulting sum is then normalized. This nuance of combining a unitless rate with a raw physical input is a simplification that is not detailed in the main explanation.
* The explanation says “uses L-BFGS-B”; in practice the multi-method cascade described in §1 is used.

---

**Take-away for contributors:** when extending the model, replicate these guardrails for any new mathematical component, and update this document if new fall-backs or constraints are introduced. 