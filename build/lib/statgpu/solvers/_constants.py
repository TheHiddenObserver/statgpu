"""Solver convergence constants and thresholds."""

_SLACK_TOLERANCE = 1e-14
_DIVERGE_COEF_NORM_CAP = 100.0
_DIVERGE_OBJ_RATIO = 100.0
_DIVERGE_OBJ_ABS = 10.0
_BB_RESTART_DOT_TOL = 1e-14
_LIPSCHITZ_FLOOR = 1e-30
_LIPSCHITZ_SAFETY_LOGISTIC_CV = 2.0

# Gradient clipping thresholds (used by fista, fista_bb, fista_lla, _array_ops)
# gmax = max(coef_norm * _GRAD_CLIP_COEF_FACTOR + _GRAD_CLIP_ABS_FLOOR, _GRAD_CLIP_MAX)
_GRAD_CLIP_COEF_FACTOR = 10.0   # scales with coefficient magnitude
_GRAD_CLIP_ABS_FLOOR = 1e3      # minimum gradient cap (prevents zero-cap at coef=0)
_GRAD_CLIP_MAX = 1e4             # absolute maximum gradient cap
