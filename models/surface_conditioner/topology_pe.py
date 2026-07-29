# -*- coding: utf-8 -*-
"""Pluggable topology positional encodings over the face-adjacency graph.

Variants: 'none' | 'random_walk' (default). Both are functions of graph
structure only -> permutation equivariant and rigid/scale invariant by
construction. (A spectral/laplacian variant lived here during ablations; it
is not permutation-equivariant under degenerate spectra and was retired.)
"""
import torch
import torch.nn as nn


def _adjacency(edges, num_faces, device):
    A = torch.zeros(num_faces, num_faces, device=device)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = 1.0
    return A


def random_walk_pe(edges, num_faces, k=16, device="cpu"):
    """RWSE: diagonal of P^t for t=1..k, P = D^-1 A. [F,k]."""
    A = _adjacency(edges, num_faces, device)
    deg = A.sum(1, keepdim=True).clamp(min=1.0)
    P = A / deg
    out = []
    M = torch.eye(num_faces, device=device)
    for _ in range(k):
        M = M @ P
        out.append(torch.diagonal(M))
    return torch.stack(out, dim=1)


class TopologyPE(nn.Module):
    """Computes (and caches nothing -- caller may cache) a [F,k] structural
    encoding from face adjacency."""

    def __init__(self, kind="random_walk", k=16):
        super().__init__()
        assert kind in ("none", "random_walk")
        self.kind = kind
        self.k = k

    @property
    def dim(self):
        return self.k

    @torch.no_grad()
    def forward(self, graph, num_faces):
        device = graph["boundary"].device
        if self.kind == "none":
            return torch.zeros(num_faces, self.k, device=device)
        return random_walk_pe(graph["edges"], num_faces, self.k, device)
