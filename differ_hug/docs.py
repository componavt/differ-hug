"""
Documentation and system information for the ODE research platform.

This module provides documentation text in a Gradio-friendly format.
"""

from typing import Optional


def documentation_markdown() -> str:
    """
    Return documentation as Markdown text for Gradio display.
    
    Returns:
        Markdown string with documentation content
    """
    return """---
**Curve Annotations:**
- Each curve is annotated with its `idx` at the final point.
- Table shows robust curvature statistics (median, p10, p90) and path length.
- FTLE diagnostics include R² of ln(dist) fit (ftle_r2). Anomaly score combines FTLE, path length, max curvature and R² reliability.

---
**Column Descriptions:**
- `idx` — Index of the trajectory, starting from 0
- `ftle` — Finite-Time Lyapunov Exponent, computed as the slope of linear fit of ln(d(t)) vs t on a central window (25%–75% of t_eval) after clipping d(t) to a minimum (1e-12)
- `ftle_r2` — Coefficient of determination (R²) of the linear fit for FTLE, indicating reliability of the ftle estimate
- `amp` — Amplitude (max−min of radial distance √(x²+y²))
- `final_dist` — Final distance between the original trajectory and its companion trajectory with tiny perturbation
- `hurst` — Hurst exponent using the rescaled range (R/S) method calculated for x(t) and y(t) and averaged
- `curv_rad_mn` — Mean curvature radius computed from 1/κ(t) where κ(t) is the curvature
- `curv_rad_med` — Median curvature radius computed from 1/κ(t) where κ(t) is the curvature
- `curv_rad_std` — Standard deviation of curvature radius
- `curv_p10` — 10th percentile of curvature radius
- `curv_p90` — 90th percentile of curvature radius
- `curv_ct_fin` — Number of finite radius samples
- `initial_x` — X coordinate of the initial point
- `initial_y` — Y coordinate of the initial point
- `path_len` — Total arclength computed as sum of Euclidean distances between consecutive points along the trajectory
- `max_kappa` — Maximum finite curvature value, useful to detect sharp bends
- `frac_high_curv` — Fraction of time points with κ(t) above a threshold (default κ > 0.1, i.e., radius < 10), measuring density of sharp bends along the trajectory
- `curv_rad_lcl_z` — Local z-score of curve median curvature radius relative to nearest neighbors in initial condition space
- `anomaly_score` — Aggregated score combining robust z-scores (IQR-based) of ftle, path_len, max_kappa, and ftle_r2 (ftle + path_len + max_kappa − ftle_r2)

---
"""


def system_latex() -> str:
    """
    Return the ODE system as LaTeX string.
    
    Returns:
        LaTeX string for the gene regulatory ODE system
    """
    return r"""
$$\begin{cases}
\displaystyle \frac{dx}{dt} = \frac{K\,x^{1/\alpha}}{b^{1/\alpha} + x^{1/\alpha}} - \gamma_1\,x,\\[10pt]
\displaystyle \frac{dy}{dt} = \frac{K\,y^{1/\alpha}}{b^{1/\alpha} + y^{1/\alpha}} - \gamma_2\,y.
\end{cases}$$
"""


def get_system_description() -> str:
    """
    Return a brief description of the gene regulatory ODE system.
    
    Returns:
        Short description string
    """
    return """Gene Regulatory ODE System

This system models gene regulation with cooperative sigmoidal activation:
- dx/dt = K * x^(1/alpha) / (b^(1/alpha) + x^(1/alpha)) - gamma1 * x
- dy/dt = K * y^(1/alpha) / (b^(1/alpha) + y^(1/alpha)) - gamma2 * y

Parameters:
- alpha: Controls the steepness of the sigmoidal response (1/alpha is the Hill coefficient)
- K: Maximum activation rate
- b: Half-activation constant
- gamma1, gamma2: Decay rates for x and y

The system exhibits bistability and can show multiple stable states depending on parameters."""
