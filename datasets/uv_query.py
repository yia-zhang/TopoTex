# -*- coding: utf-8 -*-
"""UV query helpers: face adjacency and connected surface subsets.

A partial surface query addresses a connected subset of faces; the subset's
own rasterization produces its face_id / barycentric / valid_mask (see
datasets/build_uv_queries.py).
"""
from collections import deque

import numpy as np


def face_adjacency(faces):
    """Shared-edge neighbor lists (mesh faces). Returns list[list[int]]."""
    f = np.asarray(faces, dtype=np.int64)
    e2f = {}
    for fi, tri in enumerate(f):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            e2f.setdefault(e, []).append(fi)
    adj = [[] for _ in range(len(f))]
    for fs in e2f.values():
        for a in fs:
            for b in fs:
                if a != b:
                    adj[a].append(b)
    return adj


def connected_subset(adj, n_faces, frac, rng):
    """BFS patch of ~frac*n_faces from a random seed face. On meshes with
    several connected components the frontier can exhaust early; reseed on an
    unvisited face until the target is met (patch = union of a few connected
    patches). Returns int64 face indices."""
    target = max(1, int(round(n_faces * frac)))
    seen = set()
    order = rng.permutation(n_faces)
    oi = 0
    while len(seen) < target and oi < n_faces:
        while oi < n_faces and int(order[oi]) in seen:
            oi += 1
        if oi >= n_faces:
            break
        seed = int(order[oi])
        seen.add(seed)
        qd = deque([seed])
        while qd and len(seen) < target:
            for nb in adj[qd.popleft()]:
                if nb not in seen:
                    seen.add(nb)
                    qd.append(nb)
                    if len(seen) >= target:
                        break
    return np.fromiter(seen, dtype=np.int64)
