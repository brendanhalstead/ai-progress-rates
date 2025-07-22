# AI Progress Modeling - Interactive Web App

An interactive web application for exploring AI progress trajectories using nested CES production functions with real-time parameter adjustment and visualization.

## Features

### 🎛️ Interactive Controls
- **Real-time parameter adjustment** with sliders for all model parameters
- **Production function controls**: Elasticity parameters, compute weights, software shares
- **Automation anchors**: Define progress and automation milestones
- **Simulation settings**: Adjustable time ranges and initial conditions

### 📊 Dynamic Visualization  
- **Interactive Plotly dashboard** with 4 synchronized panels:
  - Cumulative progress over time
  - Automation fraction evolution
  - Instantaneous progress rates
  - Progress vs automation correlation
- **Real-time updates** as parameters change
- **Responsive design** that works on desktop and mobile

### 📁 Data Management
- **CSV file upload** for custom time series data
- **Default dataset** with exponential AI growth scenario
- **Data validation** and preview functionality
- **Export results** to CSV for further analysis

### 🔧 Parameter Estimation
- **Anchor constraint system** for parameter fitting
- **Optimization interface** to find parameters matching expert expectations
- **Constraint validation** and error handling

### 📈 Export & Analysis
- **Download plots** as high-resolution images
- **Export trajectories** as CSV files with timestamps
- **Summary statistics** for key model outputs
- **Shareable results** via URL parameters (future feature)

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
2. **Update Model**: Click "Update Model" to recompute and visualize results
3. **Upload Data**: Switch to "Data" tab to upload custom CSV files
4. **Export Results**: Use "Export CSV" to download computed trajectories

### CSV Data Format

Upload files with these columns:
```csv
time,L_HUMAN,L_AI,experiment_compute,training_compute
2019,1000000,1000,1000000,1000000
2020,1000000,3162,3162278,3162278
...
```

## Web App Architecture

### Backend (Flask)
- **API Endpoints**: 
  - `/api/compute` - Model computation with parameters
  - `/api/upload-data` - CSV file upload and parsing
  - `/api/estimate-parameters` - Parameter optimization
  - `/api/export-csv` - Results download
  - `/api/default-data` - Load sample data

### Frontend (HTML/JavaScript)
- **Bootstrap UI** with responsive design
- **Plotly.js** for interactive visualization
- **jQuery** for API communication
- **Real-time updates** without page refresh

### Key Components

#### Parameter Controls
```javascript
// Real-time parameter collection
function collectParameters() {
    return {
        rho_cognitive: parseFloat($('#rho_cognitive').val()),
        rho_progress: parseFloat($('#rho_progress').val()),
        // ... other parameters
    };
}
```

#### Model Computation
```python
# Backend model computation
@app.route('/api/compute', methods=['POST'])
def compute_model():
    params = Parameters(**request.json['parameters'])
    model = ProgressModel(params, time_series)
    times, progress = model.compute_progress_trajectory(time_range, initial_progress)
    # Return interactive Plotly figure
```

#### Interactive Plotting
```javascript
// Update plots with new results
Plotly.newPlot('plotDiv', response.plot.data, response.plot.layout, {responsive: true});
```

## Advanced Features

### Parameter Sensitivity Analysis
The web app enables rapid exploration of parameter space:
- Adjust elasticity parameters to see substitution effects
- Modify automation anchors to test different scenarios  
- Compare software vs training progress contributions

### Scenario Modeling
Create and compare different AI development scenarios:
- Conservative: Lower automation fractions, slower AI growth
- Aggressive: Rapid automation, exponential capability gains
- Balanced: Moderate assumptions with realistic constraints

### Data Integration
Import real-world data:
- Historical AI benchmarks and capabilities
- Compute trends and scaling laws
- Labor market and automation statistics

## Deployment

### Local Development
```bash
python app.py  # Development server on localhost:5000
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t ai-progress-app .
docker run -p 5000:5000 ai-progress-app
```

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export MAX_CONTENT_LENGTH=16777216  # 16MB file upload limit
```

## API Reference

### POST /api/compute
Compute model trajectory with given parameters.

**Request:**
```json
{
    "parameters": {
        "rho_cognitive": 0.5,
        "rho_progress": 0.5,
        "alpha": 0.5,
        "software_progress_share": 0.7,
        "automation_fraction_at_superhuman_coder": 0.9,  // Maximum automation fraction (upper asymptote)
        "progress_at_half_sc_automation": 50.0,  // Progress level where automation = half of maximum
        "automation_slope": 2.0,  // Slope parameter controlling transition steepness
        "progress_rate_normalization": 1.0,
        "cognitive_output_normalization": 1e-3
    },
    "time_range": [2019, 2030],
    "initial_progress": 1.0
}
```

**Response:**
```json
{
    "success": true,
    "plot": { /* Plotly figure JSON */ },
    "summary": {
        "final_progress": 156.7,
        "final_automation": 0.85,
        "avg_progress_rate": 1250000,
        "time_range": [2019, 2030]
    }
}
```

### POST /api/upload-data
Upload custom time series CSV file.

**Request:** Multipart form with 'file' field

**Response:**
```json
{
    "success": true,
    "data_summary": {
        "time_range": [2019, 2030],
        "data_points": 12,
        "preview": { /* time series data */ }
    }
}
```

### GET /api/export-csv
Download current results as CSV file.

**Response:** CSV file with columns: time, cumulative_progress, automation_fraction, progress_rate

## Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**File upload failures:**
- Check CSV format matches expected columns
- Verify file size < 16MB
- Ensure numeric data in all columns

**Parameter optimization errors:**
- Check anchor constraints are realistic
- Verify parameter bounds are reasonable
- Try simpler constraints first

### Performance Tips

- Use reasonable time ranges (10-20 years max)
- Avoid extreme parameter values
- Upload smaller CSV files for faster processing
- Close browser dev tools for better rendering performance

## Future Enhancements

- **URL parameter sharing** for reproducible results
- **Multi-scenario comparison** with side-by-side plots  
- **Advanced constraint builder** with drag-and-drop interface
- **Real-time collaboration** with shared sessions
- **Model versioning** and result history
- **Integration with external APIs** for live data feeds

## Contributing

The web app builds on the existing Python modeling framework. Key files:

- `app.py` - Flask backend with API endpoints
- `templates/index.html` - Interactive frontend
- `progress_model.py` - Core modeling logic (reused)
- `visualization.py` - Plotting utilities (reused)

When adding features:
1. Update backend API endpoints in `app.py`
2. Add frontend controls in `templates/index.html`
3. Test with various parameter combinations
4. Ensure responsive design on mobile devices