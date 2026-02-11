
import numpy as np

def pareto_filter(points):
    pts = np.array(points, dtype=float)
    keep = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if j == i:
                continue
            if np.all(q <= p) and np.any(q < p):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    return np.array(keep)

def normalize_points(points, mins, maxs, eps=1e-12):
    pts = np.array(points, dtype=float)
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)
    return (pts - mins) / (maxs - mins + eps)

def hypervolume_2d(points, ref):
    pts = np.array(points, dtype=float)
    ref = np.array(ref, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for x, y in pts[::-1]:
        width = prev_x - x
        height = ref[1] - y
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x
    return float(hv)

def igd(approx_set, reference_set):
    A = np.array(approx_set, dtype=float)
    R = np.array(reference_set, dtype=float)
    dists = []
    for r in R:
        d = np.min(np.linalg.norm(A - r, axis=1))
        dists.append(d)
    return float(np.mean(dists))
