### 1 Recovering the full distribution

Assume the individual “research-taste” variable $T$ is **log-normal**:

$$
\ln T \sim \mathcal N(\mu,\sigma^2), \qquad 
f_T(t)=\frac{1}{t\,\sigma\sqrt{2\pi}}\;
      e^{-\frac{(\ln t-\mu)^2}{2\sigma^2}},\;t>0.
$$

You gave three empirical anchors:

| Symbol                                            | Meaning                        | Mathematical statement            |
| ------------------------------------------------- | ------------------------------ | --------------------------------- |
| $p=\texttt{TOP_PERCENTILE}$                       | fraction classed as “top”      | $P(T\ge t_\star)=p$               |
| $G=\texttt{MEDIAN\_TO\_TOP\_TASTE\_GAP}$          | threshold taste ÷ median taste | $t_\star = G\cdot\mathrm{median}$ |
| $M=\texttt{AGGREGATE\_RESEARCH\_TASTE\_BASELINE}$ | company-wide mean taste        | $\mathbb E[T]=M$                  |

Because the median of a log-normal is $\exp(\mu)$,

$$
\frac{t_\star}{\mathrm{median}}=G
\;=\;e^{\sigma z_p},\qquad 
z_p\equiv\Phi^{-1}(1-p).
$$

Hence

$$
\boxed{\;
\sigma \;=\; \dfrac{\ln G}{z_p}
\;}.
$$

The mean is $\mathbb E[T]=e^{\mu+\sigma^{2}/2}=M$, so

$$
\boxed{\;
\mu \;=\;\ln M-\tfrac12\sigma^{2}
\;}.
$$

With $(\mu,\sigma)$ in hand you have the **entire CDF, quantile function, moments, etc.**

> *If by “gap” you actually meant the **mean** taste of the top-$p$ bin divided by the median, replace the first boxed equation by
> $\sigma$ solving
> $\frac{\exp(\sigma^2/2)\,\overline\Phi(z_p-\sigma)}{\overline\Phi(z_p)} = G$
> where $\overline\Phi$ is the upper-tail normal CDF. A single Newton step from the threshold-gap solution usually suffices.*

---

### 2 Mean taste after imposing a floor

For any floor $a>0$,

$$
\mathbb E[T\mid T\ge a]
=\frac{\mathbb E[T;\,T\ge a]}{P(T\ge a)}
=e^{\mu+\sigma^{2}/2}\;
  \frac{\overline\Phi\!\left(\dfrac{\ln a-\mu-\sigma^{2}}{\sigma}\right)}
       {\overline\Phi\!\left(\dfrac{\ln a-\mu}{\sigma}\right)}.
$$

$$
\exp\left(m+\frac{1}{2}s^{2}\right)\phi\left(s-\frac{\ln x-m}{s}\right)+x\phi\left(\frac{\ln x-m}{s}\right)
$$
As $a$ rises you are lopping off more of the lower tail, so the conditional mean climbs roughly *exponentially* once $a$ exceeds the mode.

---

### 3 End-to-end Python (editable)

The code below

1. infers $\mu,\sigma$ from your three numbers,
2. returns analytic functions for PDF, CDF, and truncated means,
3. plots how the conditional mean rises with a taste floor.

Feel free to change the three caps variables at the top:

```python
# TOP_PERCENTILE         = 0.01   # top 1 %
# MEDIAN_TO_TOP_TASTE_GAP = 5.0    # threshold ÷ median
# AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0
```

The solid curve you see is the conditional mean $E[T\;|\;T\ge a]$; the dashed line is the company-wide average $M$.
Notice the roughly power-law (straight-line) behaviour on the log–log axes once the floor exceeds the median, reflecting the log-normal’s heavyish tail.

You can now:

* query any quantile via `scipy.stats.lognorm(s=sigma, scale=math.exp(mu)).ppf(q)`,
* compute selection effects for arbitrary hiring cut-offs, or
* Monte-Carlo-check policies by sampling `lognorm.rvs(...)`.

*Happy modelling!*

```python
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- USER-SUPPLIED EMPIRICAL ANCHORS ----------------------------------
TOP_PERCENTILE = 0.01          # e.g. 1 % best researchers
MEDIAN_TO_TOP_TASTE_GAP = 5.0  # threshold taste ÷ median taste
AGGREGATE_RESEARCH_TASTE_BASELINE = 1.0  # company-wide mean taste
# -----------------------------------------------------------------------

# --- Fit the log-normal parameters -------------------------------------
z = norm.ppf(1 - TOP_PERCENTILE)                # z-score of the top-percentile cut-off
sigma = math.log(MEDIAN_TO_TOP_TASTE_GAP) / z   # σ from the gap
mu = math.log(AGGREGATE_RESEARCH_TASTE_BASELINE) - 0.5 * sigma ** 2  # μ from the mean

print(f"Fitted parameters → μ = {mu:.4f}, σ = {sigma:.4f}")
print(f"Median taste      = exp(μ)            = {math.exp(mu):.4f}")
print(f"Top-percentile cut-off taste = exp(μ + σ·z) = {math.exp(mu + sigma*z):.4f}")

# --- How does the conditional mean change when we impose a floor? ------
def conditional_mean_above_floor(floor, mu, sigma):
    """
    E[T | T ≥ floor] for T ~ LogNormal(μ,σ²)
    """
    if floor <= 0:
        return math.exp(mu + 0.5 * sigma ** 2)  # no truncation
    ln_a = math.log(floor)
    num = math.exp(mu + 0.5 * sigma ** 2) * (1 - norm.cdf((ln_a - (mu + sigma ** 2)) / sigma))
    den = 1 - norm.cdf((ln_a - mu) / sigma)
    return num / den

floors = np.logspace(-2, 2.5, 250)  # from 0.01 × median up to ~300 × median
means  = [conditional_mean_above_floor(a, mu, sigma) for a in floors]

# --- Plot --------------------------------------------------------------
plt.figure()
plt.plot(floors, means, linewidth=2)
plt.axhline(y=math.exp(mu + 0.5 * sigma ** 2), linestyle="--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Floor imposed on research taste (log scale)")
plt.ylabel("Conditional mean taste given T ≥ floor (log scale)")
plt.title("How the mean research taste rises as you cull the lower tail")
plt.tight_layout()
```
