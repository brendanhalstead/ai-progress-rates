# Corrected analysis ( **clip-and-keep-everyone** version)

Below is a self-contained write-up that (i) recovers the log-normal research-taste distribution from three empirical anchors, and (ii) shows how the **company-wide mean changes when every taste draw below a floor $F$ is *lifted up* to the floor** (no one is discarded).

---

## 1  Recover the full distribution

Assume individual research taste $T$ is log-normal:

$$
\ln T \sim \mathcal N(\mu,\sigma^{2}),\qquad
f_T(t)=\frac{1}{t\,\sigma\sqrt{2\pi}}\;
        e^{-\frac{(\ln t-\mu)^{2}}{2\sigma^{2}}},\;t>0.
$$

You supplied three empirical anchors:

| Symbol | Meaning                        | Mathematical statement             |
| ------ | ------------------------------ | ---------------------------------- |
| $p$    | fraction classed as “top”      | $P(T\ge t_\star)=p$                |
| $G$    | threshold taste / median taste | $t_\star = G\times\mathrm{median}$ |
| $M$    | company-wide mean taste        | $\mathbb E[T]=M$                   |

Because $\mathrm{median}=e^{\mu}$ and the top-$p$ threshold satisfies $t_\star = e^{\mu+\sigma z_p}$ with $z_p=\Phi^{-1}(1-p)$,

$$
\boxed{\;
\sigma = \frac{\ln G}{z_p}
\;},\qquad
\boxed{\;
\mu   = \ln M-\tfrac12\sigma^{2}
\;}.
$$

(*If “gap” refers to the **mean** taste of the top-$p$ cohort divided by the median, use the tail-mean equation instead; see the previous note.*)

---

## 2  Mean research taste after **clipping** at a floor $F$

Define the clipped variable $Y=\max(T,F)$.  Write

$$
a=\frac{\ln F-\mu}{\sigma},\qquad
\Phi = \text{standard-normal CDF}.
$$

**Expectation decomposition**

$$
\begin{aligned}
\mathbb E[Y]
  &= \underbrace{\mathbb E[T;\,T\ge F]}_{\text{upper part unchanged}}
     + \underbrace{\mathbb E[F;\,T<F]}_{\text{lower tail lifted}} \\[2mm]
  &= e^{\mu+\sigma^{2}/2}\,
     \Phi(\sigma-a)
     \;+\; F\,\Phi(a).
\end{aligned}
$$

Hence the company-wide mean as a function of the floor $F$ is

$$
\boxed{\;
\displaystyle
\mathbb E[\max(T,F)]
  = F\,\Phi(a)
    +e^{\mu+\sigma^{2}/2}\,
     \Phi(\sigma-a),
\qquad
a=\frac{\ln F-\mu}{\sigma}.
\;}
$$

### Sanity checks

* **$F\to0$** $\Phi(a)\to0$, recover the original mean $e^{\mu+\sigma^{2}/2}=M$.
* **$F\to\infty$** $\Phi(a)\to1$, $\Phi(\sigma-a)\to0$, mean $\to F$.

---

## 3  End-to-end Python (editable)

```python
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- USER INPUT -------------------------------------------------------
TOP_PERCENTILE               = 0.01   # top 1 %
MEDIAN_TO_TOP_TASTE_GAP      = 3.25   # threshold ÷ median taste
AGGREGATE_RESEARCH_TASTE_MEAN = 1.0   # company-wide mean taste
# -----------------------------------------------------------------------

# --- Fit log-normal parameters ----------------------------------------
z      = norm.ppf(1 - TOP_PERCENTILE)
sigma  = math.log(MEDIAN_TO_TOP_TASTE_GAP) / z
mu     = math.log(AGGREGATE_RESEARCH_TASTE_MEAN) - 0.5 * sigma**2
overall_mean = math.exp(mu + 0.5 * sigma**2)          # should equal M

print(f"μ = {mu:.6f},  σ = {sigma:.6f},  mean = {overall_mean:.4f}")

# --- Clipped mean ------------------------------------------------------
def mean_after_clipping(F, mu=mu, sigma=sigma):
    """E[max(T, F)] for T ~ LogNormal(mu, sigma^2)."""
    a = (math.log(F) - mu) / sigma
    upper = math.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - a)
    lower = F * norm.cdf(a)
    return upper + lower

# --- Plot mean vs. floor ----------------------------------------------
floors = np.linspace(0.1, 2.5, 300)          # adjust range as desired
means  = [mean_after_clipping(F) for F in floors]

plt.figure()
plt.plot(floors, means, label="E[max(T, F)]", lw=2)
plt.axhline(overall_mean, ls="--", label="Original mean")
plt.xlabel("Floor F")
plt.ylabel("Company-wide mean after clipping")
plt.title("Effect of a hard floor on mean research taste")
plt.legend()
plt.tight_layout()
plt.show()
```

*Edit the three caps variables at the top to suit different scenarios.*
The function `mean_after_clipping` gives the closed-form expectation
and the plot visualises how aggressively the mean rises as you increase the floor while **keeping every employee**.
