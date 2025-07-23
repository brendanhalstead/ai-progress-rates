# AI Progress Model Explanation

This document provides a detailed explanation of the AI progress model implemented in `progress_model.py`. The model simulates the trajectory of AI development by capturing the interplay between AI capabilities, research inputs (labor and compute), and the automation of cognitive tasks.

## 1. Model Overview

The model is built around a core system of coupled ordinary differential equations (ODEs) that describe the trajectory of AI development. The model's dynamics emerge from several interconnected feedback loops:

1.  **Inputs**: The model takes time-series data for human labor, AI labor (in human-equivalents), and compute resources (for experiments and training).
2.  **Production Functions**: A series of nested Constant Elasticity of Substitution (CES) production functions combine these inputs to produce rates of change.
3.  **Research Stock**: A new state variable, "research stock," represents the accumulated knowledge and algorithmic sophistication. This stock grows based on research inputs (cognitive labor and experiment compute).
4.  **Software Progress**: The rate of "software progress" is no longer a direct output of a production function. Instead, it is derived from the growth of the research stock, modeling progress as the rate of *improvement relative to* the existing knowledge base.
5.  **Automation Feedback Loop**: As cumulative AI progress (`P`) accumulates, it enables greater automation of cognitive tasks. This increased automation, in turn, amplifies the effective labor input, which accelerates the growth of both research stock and overall progress.
6.  **Integration**: The model integrates the instantaneous rates over time to calculate the two core state variables: cumulative AI progress (`P`) and cumulative research stock (`RS`).

The model's dynamics are governed by a set of parameters that define the relationships between its components. These parameters can be estimated by constraining the model to match certain "anchor points"—plausible real-world observations or expert judgments.

## 2. Core Components and Equations

The model is composed of several computational steps that lead to the overall progress rate.

### 2.1. Cognitive Output (C)

The first step is to calculate the total effective "cognitive output," which is the combined labor contribution from humans and AI systems. The model uses a CES production function where the automation fraction `a` affects the weighting of each labor type. The formula is:

\[
C = N_{cognitive} \cdot \left( a^{1-\rho_{cog}} L_{AI}^{\rho_{cog}} + (1-a)^{1-\rho_{cog}} L_{human}^{\rho_{cog}} \right)^{1/\rho_{cog}}
\]

This formulation can be interpreted as a standard CES function applied to *effective* labor inputs, where the effective AI labor is \(L_{AI}/a\) and effective human labor is \(L_{human}/(1-a)\).

Where:
-   \( C \): Total cognitive output.
-   \( a \): The fraction of cognitive work that is automated (see Section 2.5).
-   \( L_{AI} \): The supply of AI labor (in human-equivalent units).
-   \( L_{human} \): The supply of human labor.
-   \( \rho_{cog} \): The elasticity of substitution parameter for cognitive work.
    -   If \( \rho_{cog} \to 1 \), AI and humans are perfect substitutes, and the formula simplifies to \( C \propto L_{AI} + L_{human} \).
    -   If \( \rho_{cog} \to 0 \), the function approaches a Cobb-Douglas form: \( C \propto \left(\frac{L_{AI}}{a}\right)^a \cdot \left(\frac{L_{human}}{1-a}\right)^{1-a} \).
    -   If \( \rho_{cog} \to -\infty \), they are perfect complements (Leontief): \( C \propto \min\left(\frac{L_{AI}}{a}, \frac{L_{human}}{1-a}\right) \).
-   \( N_{cognitive} \): A normalization constant (`cognitive_output_normalization`).

### 2.2. Research Stock Growth Rate (RS')

The cognitive output is then combined with compute dedicated to experiments (e.g., running simulations, testing new algorithms) to generate the growth rate of the "research stock." This stock represents the accumulation of scientific and technical knowledge.

\[
RS' = \left( \alpha \cdot E^{\rho_{prog}} + (1 - \alpha) \cdot C^{\rho_{prog}} \right)^{1/\rho_{prog}}
\]

Where:
-   \( RS' \): The instantaneous rate of change of the research stock.
-   \( E \): The amount of compute used for experiments (`experiment_compute`).
-   \( C \): The cognitive output from the previous step.
-   \( \alpha \): A parameter determining the relative importance of experiment compute vs. cognitive output.
-   \( \rho_{prog} \): The elasticity of substitution between compute and cognitive work for research.

### 2.3. Software Progress Rate (S)

The software progress rate is now dynamically calculated based on the state of the research stock. It is defined as the growth rate of the stock, normalized by the stock itself and the initial conditions. This formulation captures the idea that as the knowledge base (`RS`) grows, a larger absolute increase in knowledge (`RS'`) is required to achieve the same relative improvement.

\[
S(t) = \frac{RS(0) \cdot RS'(t)}{RS'(0) \cdot RS(t)}
\]

Where:
-   \( S(t) \): The software progress rate at time \(t\).
-   \( RS(t) \): The research stock at time \(t\).
-   \( RS'(t) \): The growth rate of research stock at time \(t\).
-   \( RS(0), RS'(0) \): The initial values of the research stock and its growth rate, used for normalization.

### 2.4. Overall Progress Rate (R)

The overall rate of AI progress is a weighted combination of the software progress rate and the amount of compute dedicated to training models. This captures the two primary drivers of AI advancement: better algorithms (software) and larger models trained on more data (training).

\[
R = N_{rate} \cdot \left( s \cdot S + (1 - s) \cdot T \right)
\]

Where:
-   \( R \): The instantaneous overall progress rate.
-   \( S \): The software progress rate from the previous step.
-   \( T \): The amount of compute used for training (`training_compute`).
-   \( s \): The share parameter (`software_progress_share`) that weights the contribution of software progress relative to training compute.
-   \( N_{rate} \): A normalization constant (`progress_rate_normalization`) calibrated to set the initial progress rate to 1.0.

### 2.5. Automation-Progress Feedback Loop

This is the critical feedback mechanism in the model. The fraction of cognitive work that can be automated is not fixed; it increases as cumulative AI progress grows. This is modeled using a generalized sigmoid function, which ensures the automation fraction (`a`) smoothly increases from a low base to an upper limit.

\[
a(P) = \frac{L}{1 + e^{-k(P - P_{mid})}}
\]

Where:
-   \( a(P) \): The automation fraction as a function of cumulative progress \( P \).
-   \( P \): The cumulative AI progress (the integral of \( R \) over time).
-   \( L \): The maximum automation fraction achievable (`automation_fraction_at_superhuman_coder`).
-   \( P_{mid} \): The level of cumulative progress at which automation reaches half of its maximum value, L/2 (`progress_at_half_sc_automation`).
-   \( k \): The growth rate or steepness of the sigmoid curve (`automation_slope`).

## 3. Integration and System Dynamics

The core of the model is a system of coupled ordinary differential equations (ODEs) that tie all the components together. The state of the system is defined by two variables: cumulative progress \( P \) and research stock \( RS \).

\[
\frac{dP}{dt} = R(t, P, RS)
\]
\[
\frac{dRS}{dt} = RS'(t, P)
\]

These equations are solved numerically.
- The progress rate \( \frac{dP}{dt} \) depends on time \( t \) (via inputs), cumulative progress \( P \) (via the automation fraction in cognitive output), and the research stock \( RS \) (via the software progress rate).
- The research stock rate \( \frac{dRS}{dt} \) depends on time \( t \) (via inputs) and cumulative progress \( P \) (via the automation fraction).

This system of coupled equations creates the model's rich, non-linear dynamics, where progress in one area feeds back to accelerate others.

## 4. Parameter Estimation

The model's behavior is sensitive to its parameters (e.g., \( \rho_{cog}, \rho_{prog}, \alpha \)). Since their true values are unknown, the script provides a mechanism to estimate them using `estimate_parameters`.

This function takes a set of **anchor constraints**—user-defined targets for the model's output under specific conditions. For example, a constraint might state: "When the automation fraction is 90% and AI labor is 1 billion, the progress rate should be 5.0."

The estimation process uses numerical optimization (specifically, the L-BFGS-B algorithm) to find a set of parameters that minimizes the squared error between the model's outputs and the targets defined by the anchor constraints. This allows the model to be calibrated against expert knowledge or empirical data. 