#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:22:58 2025

@author: alineos1
"""

from saspt import RBME, StateArray, normalize_2d
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import os, re, glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'DejaVu Sans'
rcParams['figure.dpi'] = 300
def evaluate_localization_fit(gt_csv, pred_csv, distance_thresh=1.0):
    """
    Compare ground truth and predicted localization CSVs.

    Args:
        gt_csv (str or pd.DataFrame): Ground truth CSV with ['x', 'y', 't']
        pred_csv (str or pd.DataFrame): Fitted/localization CSV with ['x', 'y', 't']
        distance_thresh (float): Max distance for a match (in pixels)

    Returns:
        dict: {
            "TP": int,
            "FP": int,
            "FN": int,
            "precision": float,
            "recall": float,
            "f1": float,
            "accuracy": float,
        }
    """
    if isinstance(gt_csv, str):
        gt = pd.read_csv(gt_csv)
    else:
        gt = gt_csv.copy()

    if isinstance(pred_csv, str):
        pred = pd.read_csv(pred_csv)
    else:
        pred = pred_csv.copy()
    pred['t'] = pred['frame']
    gt = gt[['x', 'y', 't']].copy()
    pred = pred[['x', 'y', 't']].copy()

    matched_gt_idx = set()
    matched_pred_idx = set()

    # Group by frame for fast matching
    for t in sorted(set(gt['t']).union(pred['t'])):
        gt_frame = gt[gt['t'] == t]
        pred_frame = pred[pred['t'] == t]

        if len(gt_frame) == 0 or len(pred_frame) == 0:
            continue

        tree = cKDTree(gt_frame[['x', 'y']].values)
        pred_coords = pred_frame[['x', 'y']].values

        distances, gt_indices = tree.query(pred_coords, distance_upper_bound=distance_thresh)

        for pred_i, (d, gt_i) in enumerate(zip(distances, gt_indices)):
            if np.isfinite(d) and gt_i < len(gt_frame):
                global_gt_i = gt_frame.index[gt_i]
                global_pred_i = pred_frame.index[pred_i]
                if global_gt_i not in matched_gt_idx and global_pred_i not in matched_pred_idx:
                    matched_gt_idx.add(global_gt_i)
                    matched_pred_idx.add(global_pred_i)

    TP = len(matched_pred_idx)
    FP = len(pred) - TP
    FN = len(gt) - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = TP / (len(gt) + FP) if len(gt) + FP > 0 else 0.0

    return f1

# --- Search and parse files ---
def extract_params(filename):
    pattern = (
        r"movie_mscale_(?P<mscale>\d+\.\d+)_trajs_k_(?P<k>\d+\.\d+)_"
        r"w_(?P<w>\d+)_t_(?P<t>\d+\.\d+)_ws_(?P<ws>\d+)\.csv"
    )
    match = re.search(pattern, filename)
    if match:
        return {
            "mscale": float(match.group("mscale")),
            "k": float(match.group("k")),
            "w": int(match.group("w")),
            "t": float(match.group("t")),
            "ws": int(match.group("ws")),
        }
    else:
        return None

# Directory to search for CSVs
search_dir = "/home/alineos1/Documents/movie_simu/H2B/bin10/mask_0"  # Change to your target directory
csv_files = glob.glob(os.path.join(search_dir, "movie_mscale_*.csv"))
os.chdir(search_dir)
# Containers
results = []
SA_dict = {}
settings = dict(
            likelihood_type = RBME,
            pixel_size_um = 0.108,
            frame_interval = 0.01,
            focal_depth = 0.7, 
            progress_bar = True,
            sample_size = 1e6,
            num_workers = os.cpu_count(),
            diff_coefs = np.logspace(-3.0, 2.0, 250))
for csv_file in csv_files:
    params = extract_params(os.path.basename(csv_file))
    if not params:
        continue  # Skip files not matching pattern
    mscale, k, w, t, ws = (params[key] for key in ['mscale', 'k', 'w', 't', 'ws'])
    target_csv = f'/home/alineos1/Documents/movie_simu/H2B/bin10/mask_0/trajectories_mscale_{mscale:.2f}.csv'
    f1 = evaluate_localization_fit(target_csv,csv_file)
    # dist = get_occurrence_distribution(csv_file)

    # Append to results list for DataFrame
    results.append({**params, "f1": f1})
    if f1>0.8:
        track_temp = pd.read_csv(csv_file)
        SA_temp = StateArray.from_detections(track_temp, **settings)
        
        param_key = tuple(params[k] for k in ["mscale", "k", "w", "t", "ws"])
        SA_temp.plot_occupations(out_png=f"mscale_{mscale:.2f}_k_{k:.1f}_w_{w}_t_{t:.1f}_ws_{ws}.png")
        SA_dict[param_key] = SA_temp.posterior_occs

# Convert to DataFrame
df_f1_scores = pd.DataFrame(results)
import pickle
with open('mScale_track_params_eva_results_bin10_042325.pkl','wb') as fh:
    pickle.dump([df_f1_scores, SA_dict], fh)

# # Output
# print(df_f1_scores)