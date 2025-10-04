

### How to generate timelines distributions
1. Specify parameter distributions and sampling settings in `config/sampling_config.yaml`.
2. Install the requirements in `requirements.txt`
3. Run `scripts/run_all.py`. 
4. Check `outputs/timestamp/` for plots.

### Core files
| File | Purpose |
| :---- | :---- |
| progress\_model.py | Core model: defines `TimeSeriesData`, `Parameters`, and `ProgressModel`; computes automation, cognitive output, software/overall progress, integrates ODEs, and supports parameter estimation/utilities. |
| model\_config.py | Central configuration: numerical constants, default parameter values, bounds/validation, and `PLOT_METADATA` used by the app/plots. |
 | input\_data.csv | Default input time series (time, `L_HUMAN`, `inference_compute`, experiment/training compute) used by the app and scripts. |
 | benchmark\_results.yaml | Benchmark metadata used to overlay SOTA METR p80 horizon points on horizon plots. |

### Webapp-related files
| File | Purpose |
| :---- | :---- |
| app.py | Flask web app/UI: generates single rollouts with adjustable parameters and serves Plotly visualizations; provides JSON/PNG endpoints and registers the Monte Carlo blueprint. |
 | templates/index.html | Main dashboard UI for the single-rollout app. |
 | static/styles.css | Styles for the web app. |

### Monte Carlo-related files

| File | Purpose |
| :---- | :---- |
 | monte\_carlo.py | Flask blueprint for Monte Carlo: builds default sampling configs from `model_config`, launches batch runs, tracks job status, serves artifacts, and renders `templates/monte_carlo.html`. |
 | scripts/batch\_rollout.py | CLI to sample parameter distributions, run many rollouts, and write `outputs/<timestamp>/` artifacts: `samples.jsonl`, `rollouts.jsonl`, `model_config_snapshot.*`, `metadata.json`. |
 | scripts/plot\_rollouts.py | CLI to plot SC time histogram, horizon trajectories, and horizon-at-SC distributions from a run; saves PNGs to the run directory. |
 | scripts/run\_all.py | One-command pipeline: batch rollout → plots → sensitivity; resolves the new run directory and logs locations. |
 | scripts/sensitivity\_analysis.py | Sensitivity over `rollouts.jsonl`: numeric correlations, categorical ANOVA, and permutation importance; optional plots and JSON summary. |
 | config/sampling\_config.yaml | Example sampling config for batch runs: parameter distributions, `num_rollouts`, `time_range`, `input_data`; CLI flags can override. |
 | templates/monte\_carlo.html | Web UI for configuring/launching Monte Carlo runs and viewing artifacts. |