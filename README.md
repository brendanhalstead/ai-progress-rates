

### How to generate timelines distributions
1. Specify parameter distributions and sampling settings in `config/sampling_config.yaml`.
2. Install the requirements in `requirements.txt`
3. Run `scripts/run_all.py`. 
4. Check `outputs/timestamp` for plots.

### Relevant files
| Relevant File | Purpose |
| :---- | :---- |
| progress\_model.py | Core model: defines `TimeSeriesData`, `Parameters`, and `ProgressModel`; computes automation, cognitive output, software/overall progress, integrates ODEs, and supports parameter estimation/utilities. |
| app.py | Flask web app/UI: generates single rollouts with adjustable parameters and serves Plotly visualizations; provides JSON/PNG endpoints and registers the Monte Carlo blueprint. |
| model\_config.py | Central configuration: numerical constants, default parameter values, bounds/validation, and `PLOT_METADATA` used by the app/plots. |


