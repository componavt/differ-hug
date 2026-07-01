"""
Core computation functions for the ODE research platform.

This module contains pure computational functions that preserve the
numerical logic from the original Streamlit implementation while
being reusable and testable.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd


def build_angles(num_points: int, circle_start: float, circle_end: float) -> np.ndarray:
    """
    Generate angles for initial conditions on a circle or sector.
    
    Args:
        num_points: Number of points to generate
        circle_start: Starting angle in degrees (0-360)
        circle_end: Ending angle in degrees (0-360)
        
    Returns:
        Array of angles in radians
    """
    cs_val = circle_start % 360
    ce_val = circle_end % 360
    span = (ce_val - cs_val) % 360
    
    if np.isclose(span, 0.0):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    else:
        if ce_val >= cs_val:
            degs = np.linspace(cs_val, ce_val, num_points, endpoint=False)
        else:
            span2 = (ce_val + 360) - cs_val
            degs = (cs_val + np.linspace(0, span2, num_points, endpoint=False)) % 360
        angles = np.deg2rad(degs)
    
    return angles


def build_initial_conditions(
    center_x: float, 
    center_y: float, 
    radius: float, 
    angles: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Build initial conditions from center, radius, and angles.
    
    Args:
        center_x: X coordinate of circle center
        center_y: Y coordinate of circle center
        radius: Radius of the circle
        angles: Array of angles in radians
        
    Returns:
        List of (x, y) initial condition tuples
    """
    initial_conditions = []
    for angle in angles:
        x0 = center_x + radius * np.cos(angle)
        y0 = center_y + radius * np.sin(angle)
        initial_conditions.append((x0, y0))
    return initial_conditions


def gene_regulatory_rhs(
    alpha_val: float, 
    K_val: float, 
    b_val: float, 
    g1_val: float, 
    g2_val: float
):
    """
    Create the right-hand side function for the gene regulatory ODE system.
    
    The system is:
    dx/dt = (K * x^(1/alpha))/(b^(1/alpha) + x^(1/alpha)) - gamma1 * x
    dy/dt = (K * y^(1/alpha))/(b^(1/alpha) + y^(1/alpha)) - gamma2 * y
    
    Args:
        alpha_val: Alpha parameter (controls exponent)
        K_val: K parameter
        b_val: b parameter
        g1_val: gamma1 parameter
        g2_val: gamma2 parameter
        
    Returns:
        RHS function suitable for scipy.integrate.solve_ivp
    """
    def rhs(t, state):
        x, y = state
        n = 1.0 / alpha_val
        if n > 1000:
            frac_x = K_val if x > b_val else 0.0
            frac_y = K_val if y > b_val else 0.0
        else:
            x_pos, y_pos = max(x, 0.0), max(y, 0.0)
            try:
                pow_b = np.power(b_val, n)
                pow_x = np.power(x_pos, n)
                pow_y = np.power(y_pos, n)
                frac_x = (K_val * pow_x) / (pow_b + pow_x) if np.isfinite(pow_x) else (K_val if x > b_val else 0.0)
                frac_y = (K_val * pow_y) / (pow_b + pow_y) if np.isfinite(pow_y) else (K_val if y > b_val else 0.0)
            except Exception:
                frac_x, frac_y = (K_val if x > b_val else 0.0), (K_val if y > b_val else 0.0)
        return [frac_x - g1_val * x, frac_y - g2_val * y]
    
    return rhs


class SciPySolver:
    """
    Solver wrapper using scipy.integrate.solve_ivp.
    """
    DEFAULT_METHOD = 'DOP853'
    DEFAULT_RTOL = 1e-9
    DEFAULT_ATOL = 1e-9
    
    def __init__(
        self, 
        method: str = DEFAULT_METHOD, 
        rtol: float = DEFAULT_RTOL, 
        atol: float = DEFAULT_ATOL
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def solve(
        self, 
        rhs, 
        x0: Tuple[float, float], 
        t_eval: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve ODE using scipy.integrate.solve_ivp.
        
        Args:
            rhs: Right-hand side function of the ODE system
            x0: Initial conditions (x0, y0)
            t_eval: Time points to evaluate
            
        Returns:
            Tuple of (success, x_solution, y_solution)
        """
        try:
            sol = solve_ivp(
                rhs, 
                (t_eval[0], t_eval[-1]), 
                x0, 
                method=self.method,
                rtol=self.rtol, 
                atol=self.atol, 
                t_eval=t_eval
            )
            if sol.success:
                return True, sol.y[0], sol.y[1]
            else:
                return False, None, None
        except Exception:
            return False, None, None


def solve_trajectory_set(
    rhs_func,
    initial_conditions: List[Tuple[float, float]],
    t_full: np.ndarray,
    t_train: np.ndarray,
    solver_method: str = 'DOP853'
) -> Tuple[List[Dict], List[Dict]]:
    """
    Solve ODE for a set of initial conditions and compute metrics.
    
    Args:
        rhs_func: RHS function for the ODE system
        initial_conditions: List of (x0, y0) tuples
        t_full: Time array for full integration
        t_train: Time array for training interval
        solver_method: Solver method to use ('DOP853' or 'RK45')
        
    Returns:
        Tuple of (solutions, metrics) where each is a list of dicts
    """
    solutions = []
    metrics = []
    
    solver = SciPySolver(method=solver_method)
    
    for idx, (x0, y0) in enumerate(initial_conditions):
        success, x_full, y_full = solver.solve(rhs_func, (x0, y0), t_full)
        
        if not success or x_full is None or y_full is None:
            solver_fallback = SciPySolver(method='RK45')
            success, x_full, y_full = solver_fallback.solve(rhs_func, (x0, y0), t_full)
            if not success or x_full is None or y_full is None:
                continue
        
        train_success, x_train, y_train = solver.solve(rhs_func, (x0, y0), t_train)
        if not train_success:
            continue
        
        solutions.append({
            "idx": idx,
            "x": x_full,
            "y": y_full,
            "t_full": t_full,
            "x_train": x_train,
            "y_train": y_train,
            "t_train": t_train,
        })
        
        amp = float(np.max(np.sqrt(x_full * x_full + y_full * y_full)) - 
                   np.min(np.sqrt(x_full * x_full + y_full * y_full)))
        
        ftle, final_d, ftle_r2 = compute_ftle_metrics(rhs_func, x0, y0, t_full[-1], t_full, x_full, y_full)
        
        hx = hurst_rs(x_full)
        hy = hurst_rs(y_full)
        hurst_val = np.nanmean([hx, hy])
        
        cr_stats = curvature_radius_stats(x_full, y_full, t_full)
        curv_mean = cr_stats["mean"]
        curv_median = cr_stats["median"]
        curv_std = cr_stats["std"]
        
        path_len = compute_path_length(x_full, y_full)
        
        kappa_arr = cr_stats.get("kappa_array")
        kappa_vals = np.array(kappa_arr) if kappa_arr is not None else np.array([])
        kappa_vals = kappa_vals[np.isfinite(kappa_vals)] if kappa_vals.size > 0 else np.array([])
        
        max_kappa = float(np.nanmax(kappa_vals)) if kappa_vals.size > 0 else np.nan
        frac_high_curv = float(np.sum(kappa_vals > 0.1) / len(t_full)) if kappa_vals.size > 0 else np.nan
        
        metrics.append({
            "idx": idx,
            "ftle": ftle,
            "ftle_r2": ftle_r2,
            "amp": amp,
            "final_dist": final_d,
            "hurst": hurst_val,
            "curv_radius_mean": curv_mean,
            "curv_radius_median": curv_median,
            "curv_radius_std": curv_std,
            "curv_p10": cr_stats["p10"],
            "curv_p90": cr_stats["p90"],
            "curv_count_finite": cr_stats["count_finite"],
            "initial_x": float(x0),
            "initial_y": float(y0),
            "path_len": path_len,
            "max_kappa": max_kappa,
            "frac_high_curv": frac_high_curv,
        })
    
    # Compute local z-score after all metrics are collected
    local_z = compute_local_zscore(metrics)
    for i, m in enumerate(metrics):
        m["curv_radius_local_zscore"] = float(local_z[i]) if local_z[i] is not None else np.nan
    
    # Compute anomaly score
    df_temp = pd.DataFrame(metrics)
    df_temp["anomaly_score"] = compute_anomaly_score(df_temp)
    for i, m in enumerate(metrics):
        m["anomaly_score"] = float(df_temp.iloc[i]["anomaly_score"])
    
    return solutions, metrics


def compute_ftle_metrics(
    rhs,
    x0: float, 
    y0: float, 
    te: float,
    t_eval: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes FTLE (Finite-Time Lyapunov Exponent) and related metrics.
    
    Args:
        rhs: Right-hand side function of the ODE system
        x0, y0: Initial conditions
        te: End time
        t_eval: Time points array
        x, y: Solution arrays from the main trajectory
        
    Returns:
        tuple: (ftle, final_d, ftle_r2) or (np.nan, np.nan, np.nan) if computation fails
    """
    eps = 1e-6 * (1.0 + abs(x0) + abs(y0))
    xp0, yp0 = x0 + eps, y0 + 0.5 * eps
    try:
        sol_p = solve_ivp(rhs, (0, te), (xp0, yp0), method='DOP853', t_eval=t_eval)
        if sol_p.success:
            xp, yp = sol_p.y
            dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
            dist = np.where(dist <= 0, 1e-12, dist)
            final_d = float(dist[-1])
            s_idx, e_idx = int(0.25 * len(t_eval)), int(0.75 * len(t_eval))
            if e_idx > s_idx + 1:
                d_slice = dist[s_idx:e_idx]
                t_slice = t_eval[s_idx:e_idx]
                d_slice = np.clip(d_slice, 1e-12, None)
                ln_d = np.log(d_slice)
                slope, intercept = np.polyfit(t_slice, ln_d, 1)
                ftle = float(slope)
                resid = ln_d - (slope * t_slice + intercept)
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((ln_d - np.mean(ln_d)) ** 2)
                ftle_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                return ftle, final_d, ftle_r2
        return np.nan, np.nan, np.nan
    except Exception:
        return np.nan, np.nan, np.nan


def hurst_rs(ts: np.ndarray) -> float:
    """
    Compute the Hurst exponent using the Rescaled Range (R/S) method.
    
    Args:
        ts: Time series data
        
    Returns:
        float: Hurst exponent or np.nan if computation fails
    """
    x = np.array(ts, dtype=float)
    N = len(x)
    if N < 20:
        return np.nan
    x = x - np.mean(x)
    Y = np.cumsum(x)
    R = np.zeros(N)
    S = np.zeros(N)
    for n in range(10, N // 2 + 1):
        seg = x[:n]
        Yseg = Y[:n]
        Rn = np.max(Yseg) - np.min(Yseg)
        Sn = np.std(seg, ddof=0)
        if Sn > 0:
            R[n - 1] = Rn
            S[n - 1] = Sn
    valid = (S > 0) & (R > 0)
    if np.sum(valid) < 3:
        return np.nan
    rs = R[valid] / S[valid]
    ns = np.arange(1, N + 1)[valid]
    try:
        H = np.polyfit(np.log(ns), np.log(rs), 1)[0]
    except Exception:
        H = np.nan
    return float(H)


def curvature_radius_stats(
    x: np.ndarray, 
    y: np.ndarray, 
    t: np.ndarray, 
    max_radius: float = 1e6, 
    clip_inf: bool = True
) -> Dict[str, Any]:
    """
    Compute robust curvature/radius statistics for a parametric curve (x(t), y(t)).
    
    Args:
        x, y: Coordinates of the curve
        t: Parameter values
        max_radius: Maximum radius to consider
        clip_inf: Whether to clip infinite/very large radii
        
    Returns:
        dict: Dictionary containing various curvature statistics
    """
    x_t = np.gradient(x, t)
    y_t = np.gradient(y, t)
    x_tt = np.gradient(x_t, t)
    y_tt = np.gradient(y_t, t)
    denom = (x_t ** 2 + y_t ** 2) ** 1.5
    num = np.abs(x_t * y_tt - y_t * x_tt)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(denom > 0, num / denom, np.nan)
    radius = np.where(np.isfinite(kappa) & (kappa != 0), 1.0 / kappa, np.nan)
    if clip_inf:
        radius = np.where(radius > max_radius, np.nan, radius)
    finite = np.isfinite(radius)
    stats = {
        "count_total": len(radius),
        "count_finite": int(np.sum(finite)),
        "frac_finite": float(np.sum(finite) / len(radius)),
        "mean": float(np.nanmean(radius)) if np.isfinite(np.nanmean(radius)) else np.nan,
        "median": float(np.nanmedian(radius)) if np.isfinite(np.nanmedian(radius)) else np.nan,
        "p10": float(np.nanpercentile(radius, 10)) if np.isfinite(np.nanpercentile(radius, 10)) else np.nan,
        "p90": float(np.nanpercentile(radius, 90)) if np.isfinite(np.nanpercentile(radius, 90)) else np.nan,
        "std": float(np.nanstd(radius)) if np.isfinite(np.nanstd(radius)) else np.nan,
        "radius_array": radius,
        "kappa_array": (1.0 / radius),
    }
    return stats


def compute_path_length(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the total path length of a curve (x(t), y(t)).
    
    Args:
        x, y: Coordinates of the curve
        
    Returns:
        float: Total path length
    """
    dx = np.diff(x)
    dy = np.diff(y)
    seg_lengths = np.sqrt(dx * dx + dy * dy)
    return float(np.sum(seg_lengths))


def compute_local_zscore(metrics: List[Dict]) -> List[Optional[float]]:
    """
    Compute local z-score of curvature median versus nearest neighbors.
    
    Args:
        metrics: List of metric dictionaries
        
    Returns:
        List of local z-scores (or None if computation fails)
    """
    n = len(metrics)
    if n <= 1:
        return [np.nan] * n
    
    arr_init = np.array([[m["initial_x"], m["initial_y"]] for m in metrics])
    rad_meds = np.array([m["curv_radius_median"] for m in metrics])
    local_z = np.full(n, np.nan)
    
    try:
        from sklearn.neighbors import NearestNeighbors
        use_sklearn = True
    except Exception:
        use_sklearn = False
    
    if use_sklearn:
        nbrs_k = min(5, n - 1)
        nbrs = NearestNeighbors(n_neighbors=nbrs_k + 1).fit(arr_init)
        distances, indices = nbrs.kneighbors(arr_init)
        for i in range(n):
            neigh_idx = indices[i, 1:]
            neigh_vals = rad_meds[neigh_idx]
            neigh_vals = neigh_vals[np.isfinite(neigh_vals)]
            if not np.isfinite(rad_meds[i]) or len(neigh_vals) < 1:
                local_z[i] = np.nan
            else:
                mu = np.mean(neigh_vals)
                sigma = np.std(neigh_vals)
                local_z[i] = (rad_meds[i] - mu) / sigma if sigma != 0 else np.nan
    else:
        nbrs_k = min(5, n - 1)
        for i in range(n):
            dists = np.linalg.norm(arr_init - arr_init[i : i + 1], axis=1)
            order = np.argsort(dists)
            neigh_idx = order[1 : 1 + nbrs_k]
            neigh_vals = rad_meds[neigh_idx]
            neigh_vals = neigh_vals[np.isfinite(neigh_vals)]
            if not np.isfinite(rad_meds[i]) or len(neigh_vals) < 1:
                local_z[i] = np.nan
            else:
                mu = np.mean(neigh_vals)
                sigma = np.std(neigh_vals)
                local_z[i] = (rad_meds[i] - mu) / sigma if sigma != 0 else np.nan
    
    return list(local_z)


def robust_z(arr: np.ndarray) -> np.ndarray:
    """
    Compute robust z-score using median and IQR.
    
    Args:
        arr: Input array
        
    Returns:
        Array of robust z-scores
    """
    arr = np.array(arr, dtype=float)
    finite = np.isfinite(arr)
    out = np.full_like(arr, np.nan)
    if np.sum(finite) == 0:
        return out
    median = np.nanmedian(arr[finite])
    q1 = np.nanpercentile(arr[finite], 25)
    q3 = np.nanpercentile(arr[finite], 75)
    iqr = q3 - q1 if q3 - q1 != 0 else 1.0
    out[finite] = (arr[finite] - median) / iqr
    return out


def compute_anomaly_score(df) -> np.ndarray:
    """
    Compute anomaly score combining multiple indicators.
    
    Uses robust z-scores (IQR-based) to combine:
    - FTLE (higher = more anomalous)
    - path_len (higher = more anomalous)
    - max_kappa (higher = more anomalous)
    - ftle_r2 (lower = more anomalous, so we subtract)
    - hurst (higher = more anomalous)
    
    Args:
        df: DataFrame with metrics
        
    Returns:
        Array of anomaly scores
    """
    if df.empty:
        return np.array([])
    
    ftle_z = robust_z(df['ftle'].values)
    path_z = robust_z(df['path_len'].values)
    kappa_z = robust_z(df['max_kappa'].values)
    r2_z = robust_z(df['ftle_r2'].fillna(0).values)
    hurst_z = robust_z(df['hurst'].fillna(0).values)
    
    score_arr = ftle_z + path_z + kappa_z - r2_z + hurst_z
    return score_arr


def compute_shadowing_diagnostics(
    solutions: List[Dict],
    epsilon_threshold: float = 1e-3
) -> Dict[str, Any]:
    """
    Compute shadowing diagnostics comparing trajectory solutions.
    
    Args:
        solutions: List of solution dictionaries (each with 'x', 'y', 't_full')
        epsilon_threshold: Threshold for shadowing breakdown
        
    Returns:
        Dictionary with shadowing diagnostics
    """
    epsilon_t_list = []
    shadowing_times = []
    
    for sol in solutions:
        x = sol['x']
        y = sol['y']
        t_full = sol['t_full']
        
        eps = 1e-6 * (1.0 + abs(x[0]) + abs(y[0]))
        xp0, yp0 = x[0] + eps, y[0] + 0.5 * eps
        try:
            sol_p = solve_ivp(
                lambda t, s: [0, 0], 
                (t_full[0], t_full[-1]), 
                (xp0, yp0), 
                method='DOP853', 
                t_eval=t_full
            )
            if sol_p.success:
                xp, yp = sol_p.y
                dist = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
                epsilon_t = np.maximum.accumulate(dist)
                epsilon_t_list.append(epsilon_t)
                
                exceed_indices = np.where(epsilon_t > epsilon_threshold)[0]
                if len(exceed_indices) > 0:
                    first_exceed_idx = exceed_indices[0]
                    shadowing_times.append(t_full[first_exceed_idx])
                else:
                    shadowing_times.append(None)
        except Exception:
            pass
    
    return {
        'epsilon_t_list': epsilon_t_list,
        'shadowing_times': shadowing_times,
    }
