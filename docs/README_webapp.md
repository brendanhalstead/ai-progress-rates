# AI Progress Modeling - Interactive Web App

An interactive web application for exploring AI progress trajectories using nested CES production functions with real-time parameter adjustment and comprehensive visualization. Features advanced research stock modeling, robust parameter estimation, and extensive numerical stability safeguards.

## Features

### 🎛️ Interactive Controls
- **Real-time parameter adjustment** with sliders for all model parameters
- **Production function controls**: Elasticity parameters, compute weights, software shares
- **Automation sigmoid anchors**: Define progress thresholds and automation curves
- **Simulation settings**: Adjustable time ranges and initial conditions
- **Research stock parameters**: Dynamic research stock configuration

### 📊 Enhanced Visualization  
- **Comprehensive Plotly dashboard** with 14 synchronized visualization panels:
  - Cumulative progress trajectory over time
  - Automation fraction evolution with sigmoid curves
  - Overall progress rates with logarithmic scaling
  - Software progress rates derived from research stock dynamics
  - Cognitive output vs experiment compute (dual y-axes)
  - Progress vs automation correlation scatter plots
  - Rate component comparisons (overall vs software)
  - Cognitive output components (AI vs human contributions)
  - Input time series (human vs AI labor, training compute)
  - Research stock accumulation and growth rates
  - Human-only progress rates (counterfactual analysis)
  - Automation progress multipliers
- **Real-time updates** as parameters change with robust error handling
- **Responsive design** optimized for desktop and mobile devices
- **Logarithmic scaling** for multi-order-of-magnitude data visualization

### 📁 Enhanced Data Management
- **CSV file upload** with comprehensive validation and preview
- **Default dataset** with realistic AI development scenarios
- **Data validation** with detailed error reporting and suggestions
- **Export enhanced results** including research stock, human-only rates, and component analysis
- **Time series interpolation** for smooth model integration

### 🔧 Advanced Parameter Estimation
- **Multi-method optimization** with L-BFGS-B, TNC, and SLSQP fallbacks
- **Strategic starting points** using Latin hypercube sampling and constraint-informed initialization
- **Robust anchor constraint system** for expert knowledge integration
- **Regularization** with quartic penalties for extreme parameters
- **Constraint pre-screening** for feasibility validation
- **Comprehensive error handling** with detailed diagnostics and suggestions

### 📈 Enhanced Export & Analysis
- **Download plots** as high-resolution images with customizable formatting
- **Export comprehensive trajectories** as CSV with 11+ metrics columns
- **Summary statistics** including automation multipliers and research stock metrics
- **Real-time computation** with detailed performance monitoring
- **Error diagnostics** with parameter adjustment suggestions

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

The app will start on `http://localhost:5000`

### Basic Usage

1. **Adjust Parameters**: Use the sliders in the "Parameters" tab to modify model behavior
2. **Update Model**: Click "Update Model" to recompute and visualize results with enhanced error handling
3. **Upload Data**: Switch to "Data" tab to upload custom CSV files with validation
4. **Estimate Parameters**: Use anchor constraints to fit model to expert expectations
5. **Export Results**: Use "Export CSV" to download comprehensive trajectory data

### CSV Data Format

Upload files with these required columns:
```csv
time,L_HUMAN,L_AI,experiment_compute,training_compute
2019,1000000,1000,1000000,1000000
2020,1000000,3162,3162278,3162278
...
```

**Data Validation Features:**
- Automatic format detection and validation
- Missing value handling with interpolation options
- Outlier detection and correction suggestions
- Time series continuity validation

## Web App Architecture

### Backend (Flask)
- **Enhanced API Endpoints**: 
  - `/api/compute` - Model computation with comprehensive error handling
  - `/api/upload-data` - Advanced CSV parsing with validation
  - `/api/estimate-parameters` - Multi-method parameter optimization
  - `/api/export-csv` - Enhanced results download with 11+ metrics
  - `/api/default-data` - Load realistic sample data

### Frontend (HTML/JavaScript)
- **Bootstrap UI** with responsive design and accessibility features
- **Plotly.js** for interactive visualization with 14 synchronized panels
- **jQuery** for seamless API communication with error handling
- **Real-time updates** with debouncing and performance optimization

### Key Components

#### Enhanced Parameter Controls
```javascript
// Real-time parameter collection with validation
function collectParameters() {
    return {
        rho_cognitive: parseFloat($('#rho_cognitive').val()),
        rho_progress: parseFloat($('#rho_progress').val()),
        alpha: parseFloat($('#alpha').val()),
        software_progress_share: parseFloat($('#software_progress_share').val()),
        automation_fraction_at_superhuman_coder: parseFloat($('#automation_fraction_at_superhuman_coder').val()),
        progress_at_half_sc_automation: parseFloat($('#progress_at_half_sc_automation').val()),
        automation_slope: parseFloat($('#automation_slope').val()),
        cognitive_output_normalization: parseFloat($('#cognitive_output_normalization').val())
    };
}
```

#### Advanced Model Computation
```python
# Backend model computation with robust error handling
@app.route('/api/compute', methods=['POST'])
def compute_model():
    params = Parameters(**request.json['parameters'])
    
    # Set up model with proper normalization and validation
    params, initial_conditions = setup_model_with_normalization(time_series, params, initial_progress)
    
    model = ProgressModel(params, time_series)
    times, progress, research_stock = model.compute_progress_trajectory(time_range, initial_progress)
    
    # Comprehensive metrics calculation including research stock dynamics
    all_metrics = model.results
    
    # Return enhanced visualization with 14 panel dashboard
```

#### Enhanced Interactive Plotting
```javascript
// Update plots with comprehensive error handling
function updatePlots(response) {
    if (response.success) {
        Plotly.newPlot('plotDiv', response.plot.data, response.plot.layout, {responsive: true});
        updateSummaryStatistics(response.summary);
    } else {
        displayErrorWithSuggestions(response.error, response.suggestions);
    }
}
```

## Advanced Features

### Research Stock Modeling
The web app now includes comprehensive research stock dynamics:
- **Dynamic initial stock calculation** based on model fundamentals
- **Stock accumulation visualization** showing knowledge growth over time
- **Research stock rate tracking** with production function inputs
- **Software progress derivation** from research stock relative growth

### Robust Parameter Estimation
Advanced optimization features:
- **Multi-method cascade**: Automatic fallback between L-BFGS-B, TNC, and SLSQP
- **Strategic initialization**: Diverse starting points for global optimization
- **Constraint feasibility**: Pre-screening for mathematically valid constraints
- **Regularization penalties**: Discourages numerically unstable parameter regions

### Enhanced Error Handling
Comprehensive error management:
- **Integration failure recovery**: Multiple ODE solver strategies with Euler fallback
- **Parameter validation**: Automatic sanitization with user warnings
- **Numerical stability**: Bounds checking and overflow protection
- **User-friendly diagnostics**: Detailed error messages with correction suggestions

### Performance Monitoring
Advanced diagnostics:
- **ODE step size logging**: Integration performance analysis
- **Computation timing**: Performance benchmarking for optimization
- **Memory usage tracking**: Resource utilization monitoring
- **Convergence analysis**: Optimization progress visualization

## API Reference

### POST /api/compute
Compute model trajectory with enhanced error handling and comprehensive metrics.

**Request:**
```json
{
    "parameters": {
        "rho_cognitive": -0.2,
        "rho_progress": -0.1,
        "alpha": 0.5,
        "software_progress_share": 0.5,
        "automation_fraction_at_superhuman_coder": 0.99,
        "progress_at_half_sc_automation": 18.0,
        "automation_slope": 1.6,
        "cognitive_output_normalization": 1.0
    },
    "time_range": [2029, 2030],
    "initial_progress": 1.0
}
```

**Enhanced Response:**
```json
{
    "success": true,
    "plot": { /* Enhanced Plotly figure with 14 panels */ },
    "summary": {
        "final_progress": 156.7,
        "final_automation": 0.85,
        "avg_progress_rate": 1250000,
        "final_research_stock": 1.2e8,
        "automation_multiplier": 3.4,
        "time_range": [2029, 2030]
    }
}
```

**Error Response with Diagnostics:**
```json
{
    "success": false,
    "error": "Integration failed: numerical instability detected",
    "error_type": "integration_failure",
    "suggestions": [
        "Try more conservative elasticity parameters (closer to 0)",
        "Reduce automation slope for smoother dynamics",
        "Check initial progress value is reasonable"
    ]
}
```

### POST /api/estimate-parameters
Advanced parameter optimization with multi-method approach.

**Request:**
```json
{
    "anchors": [
        {
            "conditions": {"automation_fraction": 0.9, "L_AI": 1e9},
            "target_variable": "progress_rate",
            "target_value": 5.0,
            "weight": 1.0
        }
    ],
    "initial_parameters": { /* parameter dict */ },
    "initial_progress": 1.0,
    "fixed_params": ["progress_rate_normalization"],
    "time_range": [2029, 2030]
}
```

**Enhanced Response:**
```json
{
    "success": true,
    "estimated_parameters": { /* optimized parameters */ },
    "optimization_info": {
        "initial_objective": 12.5,
        "final_objective": 0.02,
        "improvement": 12.48,
        "method_used": "L-BFGS-B",
        "iterations": 42,
        "function_evaluations": 156
    },
    "constraint_evaluations": [
        {
            "constraint_id": 0,
            "target_value": 5.0,
            "actual_value": 4.98,
            "error": 0.02,
            "satisfaction": 0.996
        }
    ],
    "plot": { /* Updated visualization */ },
    "summary": { /* Enhanced summary statistics */ }
}
```

### GET /api/export-csv
Download comprehensive results with enhanced metrics.

**Response:** CSV file with columns:
- `time`: Time points
- `cumulative_progress`: Total AI progress
- `automation_fraction`: Cognitive task automation level
- `progress_rate`: Overall progress rate
- `software_progress_rate`: Software component rate
- `cognitive_output`: Combined human/AI cognitive output
- `research_stock`: Accumulated knowledge stock
- `research_stock_rate`: Research stock growth rate
- `human_only_progress_rate`: Counterfactual human-only rate
- `ai_labor_contribution`: AI contribution to cognitive output
- `human_labor_contribution`: Human contribution to cognitive output

## Troubleshooting

### Common Issues

**"Integration failed" errors:**
- Try more conservative parameter values (elasticities closer to 0)
- Reduce automation slope for smoother dynamics
- Check time range and initial conditions are reasonable
- Use default parameters as a stable starting point

**Parameter estimation failures:**
- Verify anchor constraints are physically plausible
- Try fewer or simpler constraints initially
- Check constraint target values are reasonable
- Use wider parameter bounds for exploration

**File upload issues:**
- Ensure CSV has required columns: time, L_HUMAN, L_AI, experiment_compute, training_compute
- Check for missing values or non-numeric data
- Verify time series is monotonically increasing
- File size must be < 16MB

**Visualization performance:**
- Close browser developer tools for better rendering
- Use reasonable time ranges (avoid excessive data points)
- Clear browser cache if plots don't update
- Check JavaScript console for error messages

### Performance Optimization

- **Time ranges**: Use 10-20 year windows for optimal performance
- **Parameter bounds**: Stay within recommended ranges for stability
- **Data quality**: Upload clean, well-formatted CSV files
- **Browser optimization**: Use modern browsers with WebGL support

## Future Enhancements

- **Multi-scenario comparison**: Side-by-side parameter exploration
- **Advanced constraint builder**: Visual constraint definition interface
- **Real-time collaboration**: Shared modeling sessions
- **Model versioning**: Parameter set history and comparison
- **Enhanced export formats**: PDF reports, PowerPoint integration
- **API documentation**: OpenAPI/Swagger specification
- **Model validation suite**: Comprehensive testing framework

## Contributing

The web app builds on a sophisticated modeling framework with extensive safeguards:

- `app.py` - Flask backend with comprehensive error handling
- `templates/index.html` - Enhanced responsive frontend
- `progress_model.py` - Robust core modeling with numerical stability
- `model_config.py` - Comprehensive configuration system
- `visualization.py` - Advanced plotting utilities

When adding features:
1. Maintain numerical stability safeguards
2. Add comprehensive error handling with user-friendly messages
3. Update both backend validation and frontend controls
4. Test across parameter ranges for robustness
5. Ensure responsive design works on mobile devices
6. Document configuration options in `model_config.py`

Perfect for research presentations, policy analysis, and AI development scenario exploration with publication-quality visualizations and robust numerical foundations.