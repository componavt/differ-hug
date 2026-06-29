---
title: Differ Hug
emoji: 📊
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Differ Hug: Interactive ODE Research Platform

A computational framework for exploring and visualizing ordinary differential equations (ODEs) in real-time.

## Research Capabilities

- **Phase Portrait Visualization**: Generate and analyze phase space trajectories for autonomous and non-autonomous systems
- **Attractor Analysis**: Investigate chaotic attractors, limit cycles, and stable/unstable manifolds
- **Vector Field Rendering**: Visualize flow fields and direction fields with customizable resolution
- **Parameter Space Exploration**: Systematically vary system parameters to observe bifurcations and qualitative changes in dynamics
- **Numerical Integration**: Multiple solver methods (RK4, LSODA) with adaptive step size control

## Technical Stack

- **Gradio**: Interactive web interface for parameter adjustment and real-time visualization
- **NumPy/SciPy**: High-performance numerical computation and ODE solvers
- **Matplotlib**: Publication-quality phase portraits and time series plots
- **Hugging Face Spaces**: Cloud-based deployment and reproducible research environment

## Usage

Access the interactive interface to:
- Select from predefined ODE systems (Lorenz, Van der Pol, Lotka-Volterra, etc.)
- Adjust initial conditions and system parameters through interactive controls
- Generate phase portraits, time series, and 3D trajectory visualizations
- Export results for further analysis

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{differ_hug,
  author = {Krizhanovsky, Andrey},
  title = {Differ Hug: Interactive ODE Research Platform},
  url = {https://huggingface.co/spaces/componavt/differ-hug},
  year = {2026}
}