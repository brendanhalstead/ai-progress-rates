# Addendum: Research Stock Formulation for Software Progress Rate

## Overview of Changes

This addendum modifies the calculation of the software progress rate in Section 2.2 of the original documentation. Instead of directly using the CES combination of experiment compute and cognitive output as the software progress rate, we introduce an intermediate state variable called **Research Stock** that accumulates over time. The software progress rate is then computed as a scaled ratio involving the research stock and its rate of change.

## Modified Equations

### Research Stock Dynamics

We introduce a new state variable, Research Stock \(RS(t)\), which represents the cumulative body of AI research knowledge at time \(t\). The rate of change of research stock is given by the CES function that previously computed software progress directly:

\[
\frac{dRS}{dt} = RS'(t) = \left( \alpha \cdot E(t)^{\rho_{prog}} + (1 - \alpha) \cdot C(t)^{\rho_{prog}} \right)^{1/\rho_{prog}}
\]

Where:
- \(RS(t)\): Research stock at time \(t\)
- \(RS'(t)\): Rate of change of research stock
- \(E(t)\): Experiment compute at time \(t\)
- \(C(t)\): Cognitive output at time \(t\)
- \(\alpha, \rho_{prog}\): Parameters as defined in the original model

### Software Progress Rate

The software progress rate \(S(t)\) is now computed using the following formula:

\[
S(t) = \frac{RS(0) \cdot RS'(t)}{RS'(0) \cdot RS(t)}
\]

Where:
- \(RS(0)\): Initial research stock (new parameter)
- \(RS'(0)\): Initial rate of research stock growth
- \(RS(t)\): Current research stock (integrated from \(RS'(t)\))
- \(RS'(t)\): Current rate of research stock growth

### Overall Progress Rate (Unchanged)

The overall progress rate remains a weighted combination:

\[
R = N_{rate} \cdot \left( s \cdot S + (1 - s) \cdot T \right)
\]

Where \(S\) is now the software progress rate calculated using the research stock formulation above.

## Implementation Implications

### Additional State Variable

The model must now track two cumulative state variables:
1. **Cumulative Progress** \(P(t)\): As before, integrated from the overall progress rate
2. **Research Stock** \(RS(t)\): New variable, integrated from the CES combination of inputs

### Modified Differential Equation System

The system now consists of two coupled ODEs:

\[
\begin{align}
\frac{dP}{dt} &= R(t, P, RS) \\
\frac{dRS}{dt} &= \left( \alpha \cdot E(t)^{\rho_{prog}} + (1 - \alpha) \cdot C(t)^{\rho_{prog}} \right)^{1/\rho_{prog}}
\end{align}
\]

### Parameter Changes

- **Removed**: `progress_at_simulation_start` (or equivalent)
- **Added**: `research_stock_at_simulation_start` (i.e., \(RS(0)\))

### Initial Conditions

At the simulation start time \(t_0\):
- \(P(t_0) = 0\) (or as specified)
- \(RS(t_0) = RS(0)\) (new parameter)
- \(RS'(0)\) must be computed from the initial values of experiment compute and cognitive output

## Interpretation

This formulation introduces a scaling mechanism for software progress that depends on:

1. **Research Productivity Ratio** \(\frac{RS'(t)}{RS'(0)}\): How fast research is accumulating now vs. initially
2. **Research Stock Ratio** \(\frac{RS(0)}{RS(t)}\): The inverse of how much the research stock has grown

The software progress rate increases when research is being produced faster than initially, but decreases as the total stock of research grows. This can capture diminishing returns to research: as the total body of knowledge expands, each unit of new research contributes proportionally less to overall progress.

## Numerical Considerations

- Both \(P(t)\) and \(RS(t)\) must be integrated simultaneously
- Care must be taken at \(t = 0\) to properly initialize \(RS'(0)\)
- The ratio formulation requires \(RS(t) > 0\) for all \(t\), which is guaranteed if \(RS(0) > 0\) and \(RS'(t) \geq 0\)