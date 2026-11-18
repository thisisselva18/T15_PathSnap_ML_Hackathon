"""Split a GPS track into segments at long time gaps or physically impossible jumps.

Owner: P3
"""
import numpy as np


def split_segments(txy, t, gap_s=60.0, max_speed_ms=70.0):
    """Return int seg_id per point. New segment when dt > gap_s or implied speed > max_speed_ms."""
    dt = np.diff(t)
    dd = np.hypot(*np.diff(txy, axis=0).T)
    brk = (dt > gap_s) | (dd / np.maximum(dt, 1e-6) > max_speed_ms)
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int32)
