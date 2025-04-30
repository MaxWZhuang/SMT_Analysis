#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:41:05 2025

@author: alineos1
"""

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm
import pickle
def extract_7x7_profiles(movie_path, localization_df, n_samples=1000):
    """
    movie_path: path to a 3D .tif movie (T, Y, X)
    localization_df: pandas DataFrame with 'x', 'y', 'frame' columns
    n_samples: number of rows to randomly sample

    Returns: list of 7x7 np.ndarray profiles
    """
    # Load movie
    movie = io.imread(movie_path)  # shape: (T, H, W)
    T, H, W = movie.shape

    # Random sample
    df = localization_df.sample(n=n_samples).copy()

    profiles = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        x = int(np.round(row['x']))
        y = int(np.round(row['y']))
        t = int(row['frame'])

        # Bounds check
        if t < 0 or t >= T:
            continue
        if y < 3 or y >= H - 3:
            continue
        if x < 3 or x >= W - 3:
            continue

        # Extract 7x7 centered patch
        patch = movie[t, y-3:y+4, x-3:x+4]
        if patch.shape == (7, 7):
            profiles.append(patch)

    return profiles
movie_path = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/v520_Halo_H2B/Dstorm/DSTORM_640_FOV_0004.tif'
localization_df = pd.read_csv(
    "/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/v520_Halo_H2B/single_mov_tracking/trimmed_200/DSTORM_640_FOV_0004_single_cell_trajs.csv"
    )  # must have 'x', 'y', 'frame'


profiles = extract_7x7_profiles(movie_path, localization_df, n_samples=1000)

from Subpixel_SMT_real_profile import fit_gaussian_2d_55, group_profiles_by_subpixel_bin_symmetry
A = group_profiles_by_subpixel_bin_symmetry(profiles, fit_func=fit_gaussian_2d_55)
A_005bin = group_profiles_by_subpixel_bin_symmetry(profiles, fit_func=fit_gaussian_2d_55, bin_size=0.05)
with open('Grouped_profiles.pkl','wb') as fh:
    pickle.dump([A, A_005bin], fh)
