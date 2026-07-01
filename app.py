"""
Gradio UI for the ODE Research Platform.

This module implements a complete Gradio interface for the gene regulatory
ODE system, including parameter controls, simulation execution, and
visualization of results.
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
from typing import Tuple, Dict, Any

from differ_hug.compute import (
    build_angles,
    build_initial_conditions,
    gene_regulatory_rhs,
    solve_trajectory_set,
    compute_local_zscore,
    compute_anomaly_score,
    compute_shadowing_diagnostics,
    SciPySolver,
)
from differ_hug.plotting import (
    make_phase_portrait_figure,
    make_time_series_figure,
    make_shadowing_figure,
    make_metrics_table_figure,
)
from differ_hug.params import parameters_to_text, text_to_parameters
from differ_hug.docs import documentation_markdown, system_latex, get_system_description


# Default parameter values
DEFAULTS = {
    "t_number": 100,
    "t_train_end": 1.0,
    "t_full_end": 3.0,
    "alpha": 0.001,
    "K": 1.0,
    "b": 1.0,
    "gamma1": 1.0,
    "gamma2": 1.0,
    "initial_radius": 0.01,
    "num_points": 12,
    "circle_start": 0,
    "circle_end": 360,
    "center_x": 1.0,
    "center_y": 1.0,
    "solver_type": "SciPy DOP853",
}

DEFAULT_PARAMS_TEXT = parameters_to_text(DEFAULTS)


def run_simulation(
    t_number: int,
    t_train_end: float,
    t_full_end: float,
    alpha: float,
    K: float,
    b: float,
    gamma1: float,
    gamma2: float,
    initial_radius: float,
    num_points: int,
    circle_start: int,
    circle_end: int,
    center_x: float,
    center_y: float,
    solver_type: str,
    show_connections: bool,
    connection_stride: int,
    selected_indices: list,
) -> Tuple[Any, Any, Any, Any, str, str, str, str]:
    """
    Run the ODE simulation and generate all outputs.
    """
    diagnostics = []
    diagnostics.append("Starting simulation...")
    
    try:
        diagnostics.append("Step 1: Building angles and initial conditions...")
        angles = build_angles(num_points, circle_start, circle_end)
        initial_conditions = build_initial_conditions(center_x, center_y, initial_radius, angles)
        diagnostics.append(f"  Generated {len(initial_conditions)} initial conditions")
        
        diagnostics.append("Step 2: Creating ODE RHS function...")
        rhs_func = gene_regulatory_rhs(alpha, K, b, gamma1, gamma2)
        
        diagnostics.append("Step 3: Setting up solver...")
        if solver_type == "SciPy DOP853":
            solver_method = "DOP853"
        else:
            solver_method = "RK45"
        
        t_full = np.linspace(0, t_full_end, t_number)
        t_train = np.linspace(0, t_train_end, max(50, t_number // 2))
        
        diagnostics.append("Step 4: Solving trajectories...")
        solutions, metrics = solve_trajectory_set(
            rhs_func, initial_conditions, t_full, t_train, solver_method
        )
        diagnostics.append(f"  Successfully solved {len(solutions)} trajectories")
        
        if not solutions:
            diagnostics.append("  ERROR: No trajectories solved successfully")
            return (
                None, None, None, None,
                "\n".join(diagnostics),
                DEFAULT_PARAMS_TEXT,
                system_latex(),
                "ERROR: No trajectories solved successfully",
                ""
            )
        
        diagnostics.append("Step 5: Computing local z-score...")
        local_z = compute_local_zscore(metrics)
        for i, m in enumerate(metrics):
            m["curv_radius_local_zscore"] = float(local_z[i]) if local_z[i] is not None else np.nan
        
        diagnostics.append("Step 6: Computing anomaly scores...")
        df_metrics = pd.DataFrame(metrics)
        df_metrics["anomaly_score"] = compute_anomaly_score(df_metrics)
        
        diagnostics.append("Step 7: Sorting by anomaly score...")
        df_metrics = df_metrics.sort_values(by="anomaly_score", ascending=False, na_position="last")
        
        diagnostics.append("Step 8: Creating figures...")
        
        phase_fig = make_phase_portrait_figure(
            solutions, selected_indices, solver_type,
            t_train_end, t_full_end, t_number,
            show_connections, connection_stride
        )
        
        time_fig = make_time_series_figure(
            solutions, selected_indices, solver_type,
            t_train_end, t_full_end, t_number
        )
        
        shadowing_fig = None
        if show_connections:
            shadowing_fig = make_shadowing_figure(
                solutions, selected_indices, solver_type,
                t_train_end, t_full_end, t_number
            )
        
        diagnostics.append("Step 9: Creating metrics table...")
        
        df_to_display = df_metrics.copy()
        column_rename_map = {
            'curv_radius_mean': 'curv_rad_mn',
            'curv_radius_median': 'curv_rad_med',
            'curv_radius_std': 'curv_rad_std',
            'curv_radius_local_zscore': 'curv_rad_lcl_z',
            'curv_count_finite': 'curv_ct_fin'
        }
        df_to_display = df_to_display.rename(columns=column_rename_map)
        metrics_html = df_to_display.style.format("{:.3f}")._repr_html_()
        
        diagnostics.append("Step 10: Preparing outputs...")
        params_text = parameters_to_text(locals())
        
        status = f"Simulation completed. Solved {len(solutions)} trajectories."
        diagnostics.append(status)
        
        diagnostics_text = "\n".join(diagnostics)
        
        return (
            phase_fig,
            time_fig,
            shadowing_fig,
            metrics_html,
            diagnostics_text,
            params_text,
            system_latex(),
            status,
            ""
        )
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        diagnostics.append(error_msg)
        return (
            None, None, None, None,
            "\n".join(diagnostics),
            DEFAULT_PARAMS_TEXT,
            system_latex(),
            error_msg,
            ""
        )


def apply_text_to_controls(
    params_text: str,
    t_number: int,
    t_train_end: float,
    t_full_end: float,
    alpha: float,
    K: float,
    b: float,
    gamma1: float,
    gamma2: float,
    initial_radius: float,
    num_points: int,
    circle_start: int,
    circle_end: int,
    center_x: float,
    center_y: float,
    solver_type: str,
) -> dict:
    """
    Parse text parameters and update control values.
    """
    parsed = text_to_parameters(params_text)
    if not parsed:
        return {
            t_number: t_number, t_train_end: t_train_end, t_full_end: t_full_end,
            alpha: alpha, K: K, b: b, gamma1: gamma1, gamma2: gamma2,
            initial_radius: initial_radius, num_points: num_points,
            circle_start: circle_start, circle_end: circle_end,
            center_x: center_x, center_y: center_y, solver_type: solver_type,
            params_text: DEFAULT_PARAMS_TEXT
        }
    
    updates = {}
    
    int_keys = {"t_number", "num_points", "circle_start", "circle_end"}
    float_keys = {
        "t_train_end", "t_full_end", "alpha", "K", "b",
        "gamma1", "gamma2", "initial_radius", "center_x", "center_y"
    }
    
    for key, val in parsed.items():
        if key in int_keys:
            try:
                updates[key] = int(val)
            except Exception:
                pass
        elif key in float_keys:
            try:
                updates[key] = float(val)
            except Exception:
                pass
        elif key == "solver_type":
            updates["solver_type"] = val
        elif key == "circle_start_end":
            cs, ce = val
            updates["circle_start"] = cs
            updates["circle_end"] = ce
    
    if "circle_start" in parsed and "circle_end" in parsed:
        cs = int(parsed["circle_start"])
        ce = int(parsed["circle_end"])
        updates["circle_start"] = cs
        updates["circle_end"] = ce
    
    if "num_points" in parsed:
        updates["selected_indices"] = list(range(min(5, int(parsed["num_points"]))))
    
    updates["params_text"] = parameters_to_text({**parsed, **updates})
    
    return updates


def read_controls_to_text(
    t_number: int,
    t_train_end: float,
    t_full_end: float,
    alpha: float,
    K: float,
    b: float,
    gamma1: float,
    gamma2: float,
    initial_radius: float,
    num_points: int,
    circle_start: int,
    circle_end: int,
    center_x: float,
    center_y: float,
    solver_type: str,
) -> str:
    """
    Read current control values and convert to text format.
    """
    params = {
        "t_number": int(t_number),
        "t_train_end": float(t_train_end),
        "t_full_end": float(t_full_end),
        "alpha": float(alpha),
        "K": float(K),
        "b": float(b),
        "gamma1": float(gamma1),
        "gamma2": float(gamma2),
        "initial_radius": float(initial_radius),
        "num_points": int(num_points),
        "circle_start": int(circle_start),
        "circle_end": int(circle_end),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "solver_type": solver_type,
    }
    return parameters_to_text(params)


def update_selected_indices(num_points: int, solutions_len: int) -> list:
    """Update selected indices based on num_points."""
    n = min(num_points, 50)
    return list(range(min(5, n)))


with gr.Blocks(title="Differ Hug: ODE Research Platform") as demo:
    gr.Markdown("# Differ Hug: ODE Research Platform")
    gr.Markdown("Interactive Gene Regulatory ODE System")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Parameters")
            
            t_number = gr.Slider(
                label="t_number (time points)", minimum=10, maximum=1000, step=10,
                value=DEFAULTS["t_number"]
            )
            t_train_end = gr.Slider(
                label="t_train_end (training end)", minimum=0.1, maximum=5.0, step=0.1,
                value=DEFAULTS["t_train_end"]
            )
            t_full_end = gr.Slider(
                label="t_full_end (full integration end)", minimum=0.1, maximum=10.0, step=0.1,
                value=DEFAULTS["t_full_end"]
            )
            
            gr.Markdown("### System Parameters")
            alpha = gr.Number(label="alpha (1/alpha exponent)", value=DEFAULTS["alpha"])
            K = gr.Slider(label="K", minimum=0.1, maximum=5.0, step=0.1, value=DEFAULTS["K"])
            b = gr.Number(label="b", value=DEFAULTS["b"])
            gamma1 = gr.Number(label="gamma1", value=DEFAULTS["gamma1"])
            gamma2 = gr.Number(label="gamma2", value=DEFAULTS["gamma2"])
            
            gr.Markdown("### Initial Conditions")
            initial_radius = gr.Number(label="Initial radius (R)", value=DEFAULTS["initial_radius"])
            num_points = gr.Slider(label="Number of trajectories", minimum=3, maximum=50, step=1,
                                  value=DEFAULTS["num_points"])
            circle_start = gr.Slider(label="Circle start (degrees)", minimum=0, maximum=360, step=1,
                                    value=DEFAULTS["circle_start"])
            circle_end = gr.Slider(label="Circle end (degrees)", minimum=0, maximum=360, step=1,
                                  value=DEFAULTS["circle_end"])
            center_x = gr.Number(label="Center X", value=DEFAULTS["center_x"])
            center_y = gr.Number(label="Center Y", value=DEFAULTS["center_y"])
            
            gr.Markdown("### Solver Options")
            solver_type = gr.Radio(
                choices=["SciPy DOP853", "SciPy RK45"],
                label="Solver Type",
                value=DEFAULTS["solver_type"]
            )
            show_connections = gr.Checkbox(
                label="Show connections (DOP853 ↔ NN)",
                value=False
            )
            connection_stride = gr.Slider(
                label="Connection stride", minimum=1, maximum=20, step=1,
                value=5
            )
            
            selected_indices = gr.CheckboxGroup(
                label="Select trajectories to display (sorted by anomaly score)",
                choices=[],
                value=[],
                interactive=True
            )
            
            run_button = gr.Button("Run Simulation", variant="primary")
            
            gr.Markdown("### Plain Text Parameters")
            params_text = gr.Textbox(
                label="Parameters (edit and apply)",
                value=DEFAULT_PARAMS_TEXT,
                lines=3,
                max_lines=10
            )
            with gr.Row():
                apply_text_btn = gr.Button("Apply text → controls")
                read_text_btn = gr.Button("Read controls → text")
            
            status_text = gr.Textbox(label="Status", lines=2)
            
        with gr.Column(scale=2):
            gr.Markdown("### Outputs")
            
            with gr.Tabs():
                with gr.TabItem("Phase Portrait"):
                    phase_fig = gr.Plot(label="Phase Portrait")
                with gr.TabItem("Time Series"):
                    time_fig = gr.Plot(label="Time Series")
                with gr.TabItem("Shadowing"):
                    shadowing_fig = gr.Plot(label="Shadowing Analysis")
                with gr.TabItem("Metrics"):
                    metrics_df = gr.HTML(label="Metrics Table")
                with gr.TabItem("Documentation"):
                    documentation = gr.Markdown(documentation_markdown())
                with gr.TabItem("System Equation"):
                    latex_eq = gr.Markdown(system_latex())
            
            diagnostics_text = gr.Textbox(label="Diagnostics", lines=10)
    
    gr.Markdown("---")
    gr.Markdown("**Notes:**")
    gr.Markdown("- Anomaly score combines FTLE, path length, max curvature and R² reliability")
    gr.Markdown("- Top 5 anomalous trajectories are selected by default")
    
    run_button.click(
        fn=run_simulation,
        inputs=[
            t_number, t_train_end, t_full_end, alpha, K, b, gamma1, gamma2,
            initial_radius, num_points, circle_start, circle_end,
            center_x, center_y, solver_type, show_connections, connection_stride,
            selected_indices
        ],
        outputs=[
            phase_fig, time_fig, shadowing_fig, metrics_df,
            diagnostics_text, params_text, latex_eq, status_text, selected_indices
        ]
    )
    
    apply_text_btn.click(
        fn=apply_text_to_controls,
        inputs=[
            params_text, t_number, t_train_end, t_full_end, alpha, K, b,
            gamma1, gamma2, initial_radius, num_points, circle_start, circle_end,
            center_x, center_y, solver_type
        ],
        outputs=[
            t_number, t_train_end, t_full_end, alpha, K, b,
            gamma1, gamma2, initial_radius, num_points,
            circle_start, circle_end, center_x, center_y, solver_type,
            params_text, selected_indices
        ]
    )
    
    read_text_btn.click(
        fn=read_controls_to_text,
        inputs=[
            t_number, t_train_end, t_full_end, alpha, K, b, gamma1, gamma2,
            initial_radius, num_points, circle_start, circle_end,
            center_x, center_y, solver_type
        ],
        outputs=params_text
    )
    
    num_points.change(
        fn=update_selected_indices,
        inputs=[num_points, gr.Number(value=0, visible=False)],
        outputs=selected_indices
    )
    
    demo.load(
        fn=lambda: (gr.update(value=DEFAULTS["t_number"]),
                   gr.update(value=DEFAULTS["t_train_end"]),
                   gr.update(value=DEFAULTS["t_full_end"]),
                   gr.update(value=DEFAULTS["alpha"]),
                   gr.update(value=DEFAULTS["K"]),
                   gr.update(value=DEFAULTS["b"]),
                   gr.update(value=DEFAULTS["gamma1"]),
                   gr.update(value=DEFAULTS["gamma2"]),
                   gr.update(value=DEFAULTS["initial_radius"]),
                   gr.update(value=DEFAULTS["num_points"]),
                   gr.update(value=DEFAULTS["circle_start"]),
                   gr.update(value=DEFAULTS["circle_end"]),
                   gr.update(value=DEFAULTS["center_x"]),
                   gr.update(value=DEFAULTS["center_y"])),
        outputs=[
            t_number, t_train_end, t_full_end, alpha, K, b,
            gamma1, gamma2, initial_radius, num_points,
            circle_start, circle_end, center_x, center_y
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
