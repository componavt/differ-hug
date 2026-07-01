---
title: Differ Hug
emoji: 📊
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Differ Hug: ODE Research Platform

An interactive platform for exploring and analyzing ordinary differential equations (ODEs), focused on gene regulatory systems and numerical diagnostics.

## Features

### ode System
- **Gene Regulatory ODE System**: Cooperative sigmoidal activation model
- **Numerical Integration**: High-precision DOP853 solver with RK45 fallback
- **Initial Conditions**: Circular/sector-based ensembles with controllable center and radius

### Diagnostics & Metrics
- **FTLE**: Finite-Time Lyapunov Exponent with R² reliability metric
- **Hurst Exponent**: Rescaled range (R/S) analysis for time series
- **Curvature Statistics**: Mean, median, std, p10, p90 of curvature radius
- **Path Length**: Total arclength of trajectories
- **Local Z-score**: Nearest-neighbor analysis of curvature metrics
- **Anomaly Score**: Combined robust z-scores for outlying trajectories

### Visualization
- **Phase Portrait**: Trajectories in phase space with annotations
- **Time Series**: x(t) and y(t) evolution over time
- **Shadowing Analysis**: Perturbation growth visualization
- **Metrics Table**: Full diagnostic output with highlighting

## Requirements

```
gradio>=4.44.1
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas (optional for extended functionality)
```

## Usage

Run the application locally:
```bash
python app.py
```

The app will be available at `http://localhost:7860`

## ODE System

The gene regulatory system:

$$
\begin{cases}
\displaystyle \frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[10pt]
\displaystyle \frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}
$$

### Parameters
- **alpha**: Controls sigmoid steepness (1/alpha = Hill coefficient)
- **K**: Maximum activation rate
- **b**: Half-activation constant
- **gamma1, gamma2**: Decay rates

## Project Structure

```
differ-hug/
├── app.py                 # Gradio UI entry point
├── differ_hug/            # Core computation package
│   ├── __init__.py
│   ├── compute.py         # Numerical computation functions
│   ├── plotting.py        # Visualization functions
│   ├── params.py          # Parameter serialization
│   └── docs.py            # Documentation helpers
└── differential_equations_streamlit_src/  # Original Streamlit code (archived)
```

## Migration Notes

This is a Gradio port of the original Streamlit implementation. Key changes:
- Replaced `streamlit` with `gradio`
- Separated computation logic from UI
- Preserved all numerical diagnostics
- Made torch-based neural network solvers optional

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{differ_hug,
  author = {Krizhanovsky, Andrey},
  title = {Differ Hug: ODE Research Platform},
  url = {https://huggingface.co/spaces/componavt/differ-hug},
  year = {2026}
}
```
