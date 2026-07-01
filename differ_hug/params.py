"""
Parameter handling for the ODE research platform.

This module provides functions to convert parameters to/from plain text
format for easy sharing and editing.
"""

from typing import Dict, Any


_ORDER = [
    "t_number",
    "t_train_end",
    "t_full_end",
    "alpha",
    "K",
    "b",
    "gamma1",
    "gamma2",
    "initial_radius",
    "num_points",
    "circle_start",
    "circle_end",
    "center_x",
    "center_y",
]


def _fmt_value(v: Any) -> str:
    """Format numbers concisely for plain text output."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    try:
        fv = float(v)
        s = ("%g" % fv)
        return s
    except Exception:
        return str(v)


def parameters_to_text(params: Dict[str, Any]) -> str:
    """
    Convert parameter dictionary to plain text format.
    
    Args:
        params: Dictionary of parameter names and values
        
    Returns:
        Plain text string with "key=value" pairs separated by semicolons
    """
    parts = []
    used = set()
    for k in _ORDER:
        if k in params:
            parts.append(f"{k}={_fmt_value(params[k])}")
            used.add(k)
    for k, v in params.items():
        if k not in used:
            parts.append(f"{k}={_fmt_value(v)}")
    return "; ".join(parts)


def text_to_parameters(text: str) -> Dict[str, Any]:
    """
    Parse plain text to parameter dictionary.
    
    Args:
        text: String with "key=value" pairs separated by semicolons
        
    Returns:
        Dictionary with parsed parameters
    """
    result: Dict[str, Any] = {}
    if not isinstance(text, str):
        return result
    raw = text.replace("\n", ";")
    for chunk in raw.split(";"):
        if "=" not in chunk:
            continue
        key, val = chunk.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if val.endswith("°"):
            val = val[:-1]
        try:
            if val.lower().startswith("0x"):
                result[key] = int(val, 16)
            else:
                iv = int(val)
                result[key] = iv
                continue
        except Exception:
            pass
        try:
            fv = float(val)
            result[key] = fv
            continue
        except Exception:
            pass
        result[key] = val
    return result
