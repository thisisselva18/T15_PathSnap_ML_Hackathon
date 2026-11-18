"""Split a track into segments at large time gaps / impossible jumps.

Owner: P3
"""

GAP_S = 60
MAX_SPEED_MS = 70


def split_segments(track_xy, t):
    """Return an int array seg_id per GPS point."""
    raise NotImplementedError
