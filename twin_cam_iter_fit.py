#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:17:35 2025

@author: alineos1
"""

import numpy as np
import pandas as pd
import os, glob
from scipy.spatial import cKDTree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

# --- Basic Utilities ---
def remove_invalid_rows(arr):
    return arr[~np.isnan(arr).any(axis=1) & ~np.isinf(arr).any(axis=1)]

def filter_isolated_points(coords, min_dist):
    tree = cKDTree(coords)
    count = tree.query_ball_tree(tree, r=min_dist)
    # Only keep points with no close neighbors (just itself)
    isolated_indices = [i for i, c in enumerate(count) if len(c) == 1]
    return coords[isolated_indices]

def match_points_predicted(camA_coords, camB_coords, projected_A, max_dist=2.0):
    treeA = cKDTree(camA_coords)
    dists, indices = treeA.query(projected_A, distance_upper_bound=max_dist)
    valid = dists < max_dist
    matchedA = camA_coords[indices[valid]]
    matchedB = camB_coords[valid]
    return matchedA, matchedB

# --- Polynomial Model ---
def fit_polynomial_model(matchedB, matchedA, degree=2, alpha=1e-2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(matchedB)
    dx = matchedA[:, 0] - matchedB[:, 0]
    dy = matchedA[:, 1] - matchedB[:, 1]
    model_dx = Ridge(alpha=alpha).fit(X_poly, dx)
    model_dy = Ridge(alpha=alpha).fit(X_poly, dy)
    return poly, model_dx, model_dy

def predict_polynomial(coordsB, poly, model_dx, model_dy):
    X_poly = poly.transform(coordsB)
    dx_pred = model_dx.predict(X_poly)
    dy_pred = model_dy.predict(X_poly)
    return coordsB + np.stack([dx_pred, dy_pred], axis=1)

# --- Neural Network Model ---
class DisplacementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def fit_nn_model(matchedB, matchedA, epochs=200, batch_size=128, lr=1e-3):
    X = torch.tensor(matchedB, dtype=torch.float32)
    y = torch.tensor(matchedA - matchedB, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DisplacementNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def predict_nn(coordsB, model):
    with torch.no_grad():
        coordsB_tensor = torch.tensor(coordsB, dtype=torch.float32)
        shift_pred = model(coordsB_tensor).numpy()
    return coordsB + shift_pred

# --- Main Iterative Registration Function ---
def iterative_registration(
    cam_a_files: List[str], cam_b_files: List[str],
    model_type: str = 'poly', iterations: int = 10, match_dist: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    fov_data = []

    # Load and prepare all FOV data
    for i in range(len(cam_a_files)):
        dfA = pd.read_csv(cam_a_files[i])
        dfB = pd.read_csv(cam_b_files[i])
        if not {'x', 'y'}.issubset(dfA.columns) or dfA.empty:
            continue
        if not {'x', 'y'}.issubset(dfB.columns) or dfB.empty:
            continue
        camA_coords = remove_invalid_rows(dfA[['y', 'x']].to_numpy())
        camB_coords = remove_invalid_rows(dfB[['y', 'x']].to_numpy())
        camA_clean = filter_isolated_points(camA_coords, min_dist=4)
        camB_clean = filter_isolated_points(camB_coords, min_dist=4)
        fov_data.append((camA_clean, camB_clean))

    # Initial matching (nearest neighbor)
    matchedA, matchedB = [], []
    for camA, camB in fov_data:
        treeA = cKDTree(camA)
        dists, indices = treeA.query(camB, distance_upper_bound=5.0)
        valid = dists < 5.0
        matchedA.append(camA[indices[valid]])
        matchedB.append(camB[valid])
    matchedA = np.vstack(matchedA)
    matchedB = np.vstack(matchedB)

    for it in range(iterations):
        print(f"--- Iteration {it+1} ---")

        # Fit model
        if model_type == 'poly':
            poly, model_dx, model_dy = fit_polynomial_model(matchedB, matchedA)
            predictor = lambda coords: predict_polynomial(coords, poly, model_dx, model_dy)
        elif model_type == 'nn':
            model = fit_nn_model(matchedB, matchedA)
            predictor = lambda coords: predict_nn(coords, model)
        else:
            raise ValueError("Invalid model_type")

        # Predict and rematch for all FOVs
        new_matchedA, new_matchedB = [], []
        for camA, camB in fov_data:
            projected = predictor(camB)
            ma, mb = match_points_predicted(camA, camB, projected, max_dist=match_dist)
            if len(ma) >= 3:
                new_matchedA.append(ma)
                new_matchedB.append(mb)

        matchedA = np.vstack(new_matchedA)
        matchedB = np.vstack(new_matchedB)
        error = np.linalg.norm(predictor(matchedB) - matchedA, axis=1)
        print(f"Mean registration error: {np.mean(error):.4f}")
        print(f"current match point number {len(matchedA)}")

    return matchedA, matchedB, predictor

import numpy as np
import matplotlib.pyplot as plt

def plot_displacement_field(predictor, width=2304, height=512, interval=10):
    """
    Plot displacement field on a grid sampled every `interval` pixels across
    an image of size (height, width). Draws lines between raw CamB grid points
    and their projected CamA coordinates via the given predictor function.
    
    Parameters:
    - predictor: function(coords) -> projected_coords; coords is an (N,2) array of [y, x].
    - width: image width in pixels.
    - height: image height in pixels.
    - interval: grid spacing in pixels.
    """
    # Create sampling grid
    xs = np.arange(0, width, interval)
    ys = np.arange(0, height, interval)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.vstack([grid_y.ravel(), grid_x.ravel()]).T  # [y, x]

    # Predict projected coordinates
    projected = predictor(coords)

    # Plot raw and projected points and connecting lines
    plt.figure(figsize=(12, 3))
    plt.scatter(coords[:, 1], coords[:, 0], s=1, color='blue')
    plt.scatter(projected[:, 1], projected[:, 0], s=1, color='red')
    for (y0, x0), (y1, x1) in zip(coords, projected):
        plt.plot([x0, x1], [y0, y1], '-', linewidth=0.5, color='gray')
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Invert y-axis for image display
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Displacement Field: CamB → CamA')
    plt.tight_layout()
    plt.show()


directory = "/home/alineos1/Downloads/06112025/2304_1024_split_640_01um_5nM/locs_t50/"
# Glob for CamA and CamB files
cam_a_files = sorted(glob.glob(os.path.join(directory, "Terraspek_beads_Iter_*_CamA*.csv")))

cam_b_files = sorted(glob.glob(os.path.join(directory, "Terraspek_beads_Iter_*_CamB*.csv")))
matchedA, matchedB, predictor_poly = iterative_registration(cam_a_files, cam_b_files, model_type = 'poly')

plot_displacement_field(predictor_poly)
matchedA_nn, matchedB_nn, predictor_nn = iterative_registration(cam_a_files, cam_b_files, model_type = 'nn')

plot_displacement_field(predictor_nn)
