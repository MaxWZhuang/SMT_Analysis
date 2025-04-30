#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:19:33 2025

@author: ziyuanc
"""
import numpy as np
import glob, os
import tifffile
from skimage import io
def analyze_binned_noise_stats(movies, masks, bin_size=2):
    """
    Computes binned column-wise noise stats for cells and background,
    and collects per-cell average intensity and X-centroid.

    Returns:
        {
            "bg": { "mean": (n_bins,), "std": (n_bins,) },
            "cell_profiles": [ {"mean": float, "x": float}, ... ]  # One per cell
        }
    """
    from collections import defaultdict
    from skimage.measure import regionprops, label

    assert len(movies) == len(masks)
    width = movies[0].shape[2]
    n_bins = width // bin_size

    bg_vals_by_bin = defaultdict(list)
    cell_profiles = []

    for movie, mask in zip(movies, masks):
        labeled = label(mask)
        props = regionprops(labeled)

        for t in range(movie.shape[0]):
            frame = movie[t, :, :]
            for bin_idx in range(n_bins):
                col_start = bin_idx * bin_size
                col_end = col_start + bin_size

                # Get all background values in this column bin
                sub_mask = mask[:, col_start:col_end]
                sub_frame = frame[:, col_start:col_end]

                bg_vals = sub_frame[sub_mask == 0]
                if bg_vals.size > 0:
                    bg_vals_by_bin[bin_idx].extend(bg_vals.tolist())

        # Now handle cell profiles
        for prop in props:
            coords = prop.coords
            cell_vals = []
            for (y, x) in coords:
                cell_vals.extend(movie[:, y, x].tolist())
            cell_profiles.append({
                "mean": np.mean(cell_vals),
                "std": np.std(cell_vals),
                "x": prop.centroid[1],
                "y": prop.centroid[0]
            })

    # Compute mean/std per bin
    bg_mean = np.zeros(n_bins)
    bg_std = np.zeros(n_bins)
    for i in range(n_bins):
        vals = bg_vals_by_bin[i]
        if vals:
            bg_mean[i] = np.mean(vals)
            bg_std[i] = np.std(vals)

    return {
        "bg": {"mean": bg_mean, "std": bg_std},
        "cell_profiles": cell_profiles,
        "bin_size": bin_size
    }
if __name__ == '__main__':
    movie_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/HEK/v520_Halo_H2B/Dstorm'
    mask_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/HEK/v520_Halo_H2B/Dstorm/single_mov_tracking/masks'
    movies = [tifffile.imread(file)[-20:,:,:] for file in sorted(glob.glob(movie_folder + '/*.tif'))[:10]]
    masks =[io.imread(file) for file in sorted(glob.glob(mask_folder + '/*_seg.tif'))[:10]]
    noises = analyze_binned_noise_stats(movies, masks)
    import pickle
    with open('H2B_calculated_bg_noise.pkl','rb') as fh:
        noises = pickle.load(fh)