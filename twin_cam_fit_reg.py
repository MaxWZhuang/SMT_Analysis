#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 19:15:46 2025

@author: alineos1
"""

import numpy as np
from scipy.spatial import cKDTree
from skimage import transform
import os, glob
import pandas as pd
def filter_isolated_points(coords, min_dist):
    """
    Removes points that have another point closer than min_dist.
    """
    tree = cKDTree(coords)
    count = tree.query_ball_tree(tree, r=min_dist)
    # Only keep points with no close neighbors (just itself)
    isolated_indices = [i for i, c in enumerate(count) if len(c) == 1]
    return coords[isolated_indices]

def match_points(camA_coords, camB_coords, max_dist):
    """
    Matches each CamB point to the nearest CamA point within max_dist.
    Returns matched (CamA, CamB) pairs.
    """
    treeA = cKDTree(camA_coords)
    dists, indices = treeA.query(camB_coords, distance_upper_bound=max_dist)
    valid = dists < max_dist
    matchedA = camA_coords[indices[valid]]
    matchedB = camB_coords[valid]
    return matchedA, matchedB


def remove_invalid_rows(arr):
    return arr[~np.isnan(arr).any(axis=1) & ~np.isinf(arr).any(axis=1)]
# Example usage:
directory = "/media/alineos1/42368BE57732CF06/Swept_Hilo/05162025/Bead_twin_cam_setup_640_split/loc_t45"
# Glob for CamA and CamB files
cam_a_files = sorted(glob.glob(os.path.join(directory, "Laser_reg_Iter_*_CamA*.csv")))
cam_b_files = sorted(glob.glob(os.path.join(directory, "Laser_reg_Iter_*_CamB*.csv")))
matched_A_stack = []
matched_B_stack = []

for i in range(len(cam_a_files)):
    try:
        dfA = pd.read_csv(cam_a_files[i])
        dfB = pd.read_csv(cam_b_files[i])
        
        # Skip if missing x/y or empty
        if not {'x', 'y'}.issubset(dfA.columns) or dfA.empty:
            print(f"Skipping {cam_a_files[i]}: missing or empty")
            continue
        if not {'x', 'y'}.issubset(dfB.columns) or dfB.empty:
            print(f"Skipping {cam_b_files[i]}: missing or empty")
            continue

        # Convert to NumPy and clean invalid rows
        camA_coords = remove_invalid_rows(dfA[['y', 'x']].to_numpy())
        camB_coords = remove_invalid_rows(dfB[['y', 'x']].to_numpy())
        camA_clean = filter_isolated_points(camA_coords, min_dist = 5)
        camB_clean = filter_isolated_points(camB_coords, min_dist = 5)
        # Estimate and collect matched points
        matchedA, matchedB = match_points(camA_clean, camB_clean, max_dist=3)
        if len(matchedA) < 3:
            print(f"Skipping pair {i}: too few matched points")
            continue

        matched_A_stack.append(matchedA)
        matched_B_stack.append(matchedB)

    except Exception as e:
        print(f"Error with pair {i}: {e}")
        continue
all_matchedA = np.vstack(matched_A_stack)
all_matchedB = np.vstack(matched_B_stack)

# Estimate final transform
tform = transform.estimate_transform('affine', all_matchedB[:, ::-1], all_matchedA[:, ::-1])
tform_poly = transform.estimate_transform('polynomial', all_matchedB[:, ::-1], all_matchedA[:, ::-1], order=2)
transformed = tform(all_matchedB[:, ::-1])[:, ::-1]
raw_error = np.linalg.norm(all_matchedB - all_matchedA, axis=1)
registered_error = np.linalg.norm(transformed - all_matchedA, axis=1)

print("Mean raw error:", np.mean(raw_error))
print("Mean transformed error:", np.mean(registered_error))

transformed_poly = tform_poly(all_matchedB[:, ::-1])[:, ::-1]
registered_poly_error = np.linalg.norm(transformed_poly - all_matchedA, axis=1)

print("Mean raw error:", np.mean(raw_error))
print("Mean poly transformed error:", np.mean(registered_poly_error))


tform_sim = transform.estimate_transform('similarity', all_matchedB[:, ::-1], all_matchedA[:, ::-1])
transformed_sim = tform_sim(all_matchedB[:, ::-1])[:, ::-1]
registered_sim_error = np.linalg.norm(transformed_sim - all_matchedA, axis=1)

print("Mean raw error:", np.mean(raw_error))
print("Mean sim transformed error:", np.mean(registered_sim_error))


from skimage.transform import PiecewiseAffineTransform
tform_piecewise = PiecewiseAffineTransform()
tform_piecewise.estimate(all_matchedB[:600, ::-1], all_matchedA[:600, ::-1])  # [x, y] order
transformed_piecewise = tform_piecewise(all_matchedB[600:, ::-1])[:, ::-1]
registered_piecewise_error = np.linalg.norm(transformed_piecewise - all_matchedA[600:,:], axis=1)

print("Mean raw error:", np.mean(raw_error))
print("Mean piecewise transformed error:", np.mean(registered_piecewise_error))




