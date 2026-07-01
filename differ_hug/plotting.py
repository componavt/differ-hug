"""
Plotting functions for the ODE research platform.

This module provides functions to create matplotlib figures
for phase portraits, time series, and shadowing analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def make_phase_portrait_figure(
    solutions: List[Dict],
    selected_indices: List[int],
    solver_type: str = "DOP853",
    t_train_end: float = 1.0,
    t_full_end: float = 3.0,
    t_number: int = 100,
    show_connections: bool = False,
    connection_stride: int = 5,
) -> plt.Figure:
    """
    Create phase portrait figure showing trajectories in phase space.
    
    Args:
        solutions: List of solution dictionaries with 'x', 'y', 't_full'
        selected_indices: List of indices to display
        solver_type: Type of solver used
        t_train_end: Training interval end time
        t_full_end: Full integration end time
        t_number: Number of time points
        show_connections: Whether to show connections between DOP853 and NN
        connection_stride: Stride for connection lines
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = ['-', '--', '-.', ':']
    colors = plt.cm.tab20.colors
    
    for m, solution_data in enumerate(solutions):
        if m not in selected_indices:
            continue
        color = colors[m % len(colors)]
        
        x_dop = solution_data["x"]
        y_dop = solution_data["y"]
        ax.plot(x_dop, y_dop, linestyle='-', color=color, linewidth=1.2,
                label=f'DOP853 traj {m}' if m == selected_indices[0] else "")
        
        ax.plot(x_dop[0], y_dop[0], 'o', color=color, markersize=4)
        ax.plot(x_dop[-1], y_dop[-1], 'x', color=color, markersize=6)
        ax.text(x_dop[-1] + 0.01, y_dop[-1] + 0.01, f"{m}", fontsize=8, color=color)
        
        if "x_nn" in solution_data and solution_data["x_nn"] is not None:
            x_nn = solution_data["x_nn"]
            y_nn = solution_data["y_nn"]
            t_full = solution_data["t_full"]
            
            ax.plot(x_nn, y_nn, linestyle='--', color=color, linewidth=1.0, alpha=0.7,
                    label=f'NN traj {m}' if m == selected_indices[0] else "")
            
            ax.plot(x_dop[0], y_dop[0], '^', color=color, markersize=8, markeredgecolor='black')
            
            if x_nn is not None and y_nn is not None:
                train_idx = np.searchsorted(t_full, t_train_end)
                if train_idx < len(x_dop):
                    ax.plot(x_nn[train_idx], y_nn[train_idx], '^', color=color, markersize=8,
                            markeredgecolor='black', markerfacecolor='none')
                    
                    ax.plot([x_dop[train_idx], x_nn[train_idx]],
                            [y_dop[train_idx], y_nn[train_idx]],
                            color=color, linewidth=1.0, alpha=0.7, linestyle=':')
            
            if show_connections:
                for i in range(0, len(x_dop), connection_stride):
                    ax.plot([x_dop[i], x_nn[i]], [y_dop[i], y_nn[i]],
                           color=color, linewidth=0.5, alpha=0.3, linestyle='-')
        
        ax.text(x_dop[-1] + 0.01, y_dop[-1] + 0.01, f"{m}", fontsize=8, color=color)
        if "x_nn" in solution_data and solution_data["x_nn"] is not None:
            if solution_data["x_nn"] is not None:
                ax.text(solution_data["x_nn"][-1] + 0.01, solution_data["y_nn"][-1] + 0.01,
                        f"{m}", fontsize=8, color=color)
    
    ax.set_title(f"Gene regulatory trajectories ({solver_type}) "
                 f"— t_train_end={t_train_end:.2f}, t_full_end={t_full_end:.2f}, t_points={t_number}")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("y(t)")
    ax.grid(True)
    
    if len(selected_indices) <= 3:
        ax.legend()
    else:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, linestyle='-', label='DOP853'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', label='NN')
        ]
        ax.legend(handles=legend_elements)
    
    return fig


def make_time_series_figure(
    solutions: List[Dict],
    selected_indices: List[int],
    solver_type: str = "DOP853",
    t_train_end: float = 1.0,
    t_full_end: float = 3.0,
    t_number: int = 100,
) -> plt.Figure:
    """
    Create time series figure showing x(t) and y(t) over time.
    
    Args:
        solutions: List of solution dictionaries
        selected_indices: List of indices to display
        solver_type: Type of solver used
        t_train_end: Training interval end time
        t_full_end: Full integration end time
        t_number: Number of time points
        
    Returns:
        matplotlib Figure object
    """
    fig, ax_ts = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab20.colors
    
    for m, solution_data in enumerate(solutions):
        if m not in selected_indices:
            continue
        color = colors[m % len(colors)]
        
        t_full = solution_data["t_full"]
        x_dop = solution_data["x"]
        y_dop = solution_data["y"]
        
        ax_ts.plot(t_full, x_dop, linestyle='-', color=color, linewidth=1.2,
                   label=f'x(t) traj {m}' if m == selected_indices[0] else "")
        ax_ts.plot(t_full, y_dop, linestyle='--', color=color, linewidth=1.2,
                   label=f'y(t) traj {m}' if m == selected_indices[0] else "")
    
    ax_ts.set_title(f"Time series ({solver_type}) — x(t) and y(t) — "
                    f"t_train_end={t_train_end:.2f}, t_full_end={t_full_end:.2f}, t_points={t_number}")
    ax_ts.set_xlabel("t")
    ax_ts.set_ylabel("x(t), y(t)")
    ax_ts.grid(True)
    ax_ts.legend()
    
    return fig


def make_shadowing_figure(
    solutions: List[Dict],
    selected_indices: List[int],
    solver_type: str = "DOP853",
    t_train_end: float = 1.0,
    t_full_end: float = 3.0,
    t_number: int = 100,
    epsilon_threshold: float = 1e-3,
) -> Optional[plt.Figure]:
    """
    Create shadowing figure showing epsilon(t) vs time.
    
    Args:
        solutions: List of solution dictionaries
        selected_indices: List of indices to display
        solver_type: Type of solver used
        t_train_end: Training interval end time
        t_full_end: Full integration end time
        t_number: Number of time points
        epsilon_threshold: Threshold for shadowing breakdown
        
    Returns:
        matplotlib Figure object or None if no shadowing data available
    """
    if not selected_indices:
        return None
    
    fig, ax_shad = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab20.colors
    
    annotated = False
    for m, solution_data in enumerate(solutions):
        if m not in selected_indices:
            continue
        color = colors[m % len(colors)]
        
        t_full = solution_data["t_full"]
        x_dop = solution_data["x"]
        y_dop = solution_data["y"]
        
        eps = 1e-6 * (1.0 + abs(x_dop[0]) + abs(y_dop[0]))
        xp0, yp0 = x_dop[0] + eps, y_dop[0] + 0.5 * eps
        try:
            from scipy.integrate import solve_ivp
            sol_p = solve_ivp(
                lambda t, s: [0, 0], 
                (t_full[0], t_full[-1]), 
                (xp0, yp0), 
                method='DOP853', 
                t_eval=t_full
            )
            if sol_p.success:
                xp, yp = sol_p.y
                dist = np.sqrt((x_dop - xp)**2 + (y_dop - yp)**2)
                epsilon_t = np.maximum.accumulate(dist)
                
                ax_shad.plot(epsilon_t, t_full, color=color, linewidth=1.2,
                            label=f'ε(t) traj {m}' if m == selected_indices[0] else "")
                
                exceed_indices = np.where(epsilon_t > epsilon_threshold)[0]
                if len(exceed_indices) > 0:
                    first_exceed_idx = exceed_indices[0]
                    shadowing_time = t_full[first_exceed_idx]
                    ax_shad.axhline(y=shadowing_time, color=color, linestyle=':', alpha=0.7,
                                   label=f't*={shadowing_time:.2f}' if not annotated else "")
                    annotated = True
        except Exception:
            pass
    
    ax_shad.axhline(y=t_train_end, color='red', linestyle='--', alpha=0.7,
                   label=f't_train_end={t_train_end}')
    
    ax_shad.set_title(f"Shadowing ({solver_type}) — ε(t) vs t — "
                      f"t_train_end={t_train_end:.2f}, t_full_end={t_full_end:.2f}, t_points={t_number}")
    ax_shad.set_xlabel("ε(t)")
    ax_shad.set_ylabel("t")
    ax_shad.grid(True)
    
    if len(selected_indices) <= 3:
        ax_shad.legend()
    else:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, linestyle='-', label='ε(t)'),
            Line2D([0], [0], color='red', lw=1, linestyle='--', label=f'train/extrap boundary')
        ]
        ax_shad.legend(handles=legend_elements)
    
    return fig


def make_metrics_table_figure(
    metrics_df,
    show_extremes: bool = True,
) -> plt.Figure:
    """
    Create a figure showing the metrics table.
    
    Args:
        metrics_df: DataFrame with metrics
        show_extremes: Whether to highlight extreme values
        
    Returns:
        matplotlib Figure object
    """
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No metrics to display", ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    column_rename_map = {
        'curv_radius_mean': 'curv_rad_mn',
        'curv_radius_median': 'curv_rad_med',
        'curv_radius_std': 'curv_rad_std',
        'curv_radius_local_zscore': 'curv_rad_lcl_z',
        'curv_count_finite': 'curv_ct_fin'
    }
    
    df_display = metrics_df.rename(columns=column_rename_map).copy()
    df_display = df_display.reset_index(drop=True)
    
    numeric_columns = df_display.select_dtypes(include=['number']).columns.tolist()
    
    if show_extremes:
        max_cols = ['idx', 'ftle', 'ftle_r2', 'amp', 'final_dist', 'hurst', 'curv_ct_fin',
                   'path_len', 'max_kappa', 'frac_high_curv', 'anomaly_score']
        min_cols = ['curv_rad_mn', 'curv_rad_med', 'curv_rad_std', 'curv_rad_lcl_z', 'curv_p10', 'curv_p90']
        
        styles = [['' for _ in range(len(df_display.columns))] for _ in range(len(df_display))]
        
        for i, col in enumerate(df_display.columns):
            if col == 'idx':
                continue
            if col in max_cols and df_display[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if not df_display[col].isna().all():
                    valid = df_display[col].dropna()
                    if len(valid) >= 2:
                        top2 = valid.nlargest(2)
                        for idx in top2.index:
                            styles[idx][i] = 'background-color: #D2691E'
                    elif len(valid) == 1:
                        styles[valid.index[0]][i] = 'background-color: #D2691E'
            elif col in min_cols and df_display[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if not df_display[col].isna().all():
                    valid = df_display[col].dropna()
                    if len(valid) >= 2:
                        bot2 = valid.nsmallest(2)
                        for idx in bot2.index:
                            styles[idx][i] = 'background-color: #00CED1'
                    elif len(valid) == 1:
                        styles[valid.index[0]][i] = 'background-color: #00CED1'
        
        df_display = df_display.style.apply(lambda x: styles, axis=None)
    
    ax.axis('off')
    
    table_str = df_display.format("{:.3f}")._repr_html_()
    ax.text(0.01, 0.99, table_str, transform=ax.transAxes, ha='left', va='top', fontsize=8)
    
    return fig
