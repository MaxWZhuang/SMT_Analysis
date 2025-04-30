#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 19:05:11 2025

@author: Ziyuan Chen
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label, find_objects, sum as ndi_sum

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import sys
import os
import glob
from skimage import io
import saspt, strobesim
from saspt import sample_detections, StateArray, RBME
from strobesim import strobe_multistate
from cell_generation import Cell
from skimage.measure import regionprops, label
from skimage.draw import rectangle
import random
import cv2
import tifffile
from collections import defaultdict
def get_subpixel_bin_and_flips(xf, yf, bin_size=0.1):
    """
    Given float position, return:
    - subpixel offset from center pixel (dy, dx)
    - corresponding flip signs (±1, ±1)
    - canonical bin in (0, 0.5)^2 grid
    """
    dy = yf - np.round(yf)
    dx = xf - np.round(xf)
    num_bins = int(np.floor(0.5 / bin_size)) 

    flip_y = -1 if dy < 0 else 1
    flip_x = -1 if dx < 0 else 1

    dy = abs(dy)
    dx = abs(dx)

    bin_y = int(np.clip(dy / bin_size, 0, num_bins-1))
    bin_x = int(np.clip(dx / bin_size, 0, num_bins-1))

    return (bin_y, bin_x), (flip_y, flip_x)

def flip_profile(profile, flip_y, flip_x):
    if flip_y == -1:
        profile = np.flipud(profile)
    if flip_x == -1:
        profile = np.fliplr(profile)
    return profile

def filter_and_crop_profiles(profiles, fit_func, center_threshold=1.0):
    """
    Filters 7x7 profiles where the maximum is within the central 3x3,
    and crops to 5x5 centered on the peak.

    Args:
        profiles (list of np.ndarray): List of 7x7 numpy arrays.

    Returns:
        List of 5x5 numpy arrays with peak at center.
    """
    filtered = []
    for p in profiles:
        if p.shape != (7, 7):
            continue  # skip invalid shapes
        try:
            fit = fit_func(p)
            if fit is None:
                continue
            fy, fx = fit[0], fit[1]
        except Exception:
            continue
        # Ensure the fit is within a safe central zone
        if abs(fy - 3.0) > center_threshold or abs(fx - 3.0) > center_threshold:
            continue
        # Round the subpixel center to get the cropping center
        y, x = int(np.round(fy)), int(np.round(fx))
        # Crop 5x5 region centered on the fit
        if 2 <= y <= 4 and 2 <= x <= 4:
            cropped = p[y - 2:y + 3, x - 2:x + 3]
            if cropped.shape == (5, 5):
                filtered.append(cropped)
    return filtered

def group_profiles_by_subpixel_bin_symmetry(profiles, fit_func, bin_size=0.1):
    """
    Groups 5x5 PSFs by subpixel center into 5x5 grid within (0, 0.5)^2.
    Projects symmetric positions to positive quadrant using flipping.

    Args:
        profiles (list): List of 5x5 numpy arrays.
        fit_func (callable): The 2D Gaussian fitting function returning (y, x, I0, bg, ...)
        bin_size (float): Grid step size, default 0.1 for 0.5/5
    Returns:
        dict: {(bin_y, bin_x): [profile, ...]} — aligned to top-right quadrant
    """
    profiles = filter_and_crop_profiles(profiles = profiles, fit_func = fit_func)
    num_bins = int(np.floor(0.5 / bin_size)) 
    grouped = defaultdict(list)
    center = 2.0  # center pixel in 5x5

    for profile in profiles:
        if profile.shape != (5, 5):
            continue
        try:
            fit = fit_func(profile)
            dy, dx = fit[0] - center, fit[1] - center
            # Flip into top-right quadrant (positive dy, dx)
            if dy < 0:
                profile = np.flipud(profile)
                dy = -dy
            if dx < 0:
                profile = np.fliplr(profile)
                dx = -dx
            # Quantize to (0, 0.5) region in 5 bins
            bin_y = int(np.clip(dy / bin_size, 0, num_bins-1))
            bin_x = int(np.clip(dx / bin_size, 0, num_bins-1))
            grouped[(bin_y, bin_x)].append(profile)
        except Exception as e:
            print("Fitting failed:", e)
    return grouped
from scipy.optimize import curve_fit
def gaussian_2d(coords, y0, x0, A, sigma, bg):
    y, x = coords
    g = A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + bg
    return g.ravel()
def fit_gaussian_2d_55(img, sigma = 1):
    """
    Fit a 2D Gaussian (non-integrated) to a 5x5 image patch.

    Returns:
        params = [y0, x0, A, sigma, bg]
    """
    if img.shape == (5, 5): #"Input image must be 5x5"
        y = np.arange(img.shape[0])
        x = np.arange(img.shape[1])
        Y, X = np.meshgrid(y, x, indexing='ij')
        guess = [2.0, 2.0, img.max() - np.median(img), sigma, np.median(img)]
        try:
            popt, _ = curve_fit(gaussian_2d,(Y, X),img.ravel(),p0=guess,bounds=([1.0, 1.0, 0, 0.5, 0],[3.0, 3.0, np.inf, 2.0, np.inf]))
            return popt  # [y0, x0, A, sigma, bg]
        except RuntimeError:
            return None
    elif img.shape == (7,7):
        y = np.arange(img.shape[0])
        x = np.arange(img.shape[1])
        Y, X = np.meshgrid(y, x, indexing='ij')
        guess = [3.0, 3.0, img.max() - np.median(img), sigma, np.median(img)]
        try:
            popt, _ = curve_fit(gaussian_2d,(Y, X),img.ravel(),p0=guess,bounds=([1.0, 1.0, 0, 0.5, 0],[5.0, 5.0, np.inf, 2.0, np.inf]))
            return popt  # [y0, x0, A, sigma, bg]
        except RuntimeError:
            return None
class Read_cell_ols:
    def __init__(self, trajectories, cell_mask, bind_profiles, noise, density_per_cell=10, max_frames=10, bin_size = 0.1, random_seed=None):
        """
        Args:
            trajectories (pd.DataFrame): Must include ['x', 'y', 'frame', 'trajectory'].
            cell_mask (2D np.ndarray): Labeled mask defining distinct cells.
            molecule_profiles (list of 7x7 np.ndarrays): Single-molecule images.
            noise (dict): Column-wise 'cell' and 'bg' noise {mean, std}, each (width,) arrays.
            density_per_cell (int): Number of trajectories to assign per cell.
            max_frames (int): Total number of frames to simulate.
        """
        self.trajectories = trajectories
        self.trajectory_groups = [group for _, group in trajectories.groupby('trajectory')]
        self.cell_mask = cell_mask
        self.molecule_profiles = bind_profiles
        self.density_per_cell = density_per_cell
        self.noise = noise  # noise["cell"]["mean"], noise["bg"]["std"], etc.
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        self.cell_regions = regionprops(label(cell_mask))
        self.frame_count = max_frames
        self.height, self.width = cell_mask.shape
        self.movie_shape = (self.height, self.width, self.frame_count)
        self.movie = np.zeros(self.movie_shape, dtype=np.float32)
        self.inserted_trajectories = [] 
        self._apply_columnwise_noise()
        self.bin_size = bin_size
    
    def _apply_columnwise_noise(self):
        """
        Optimized background generation with binned columns + cell patch replacement.
        """
        n_bins = len(self.noise["bg"]["mean"])
        bin_size = self.noise["bin_size"]
    
        # Step 1: Global background
        for bin_idx in range(n_bins):
            col_start = bin_idx * bin_size
            col_end = col_start + bin_size
            mean = self.noise["bg"]["mean"][bin_idx]
            std = self.noise["bg"]["std"][bin_idx]
            self.movie[:, col_start:col_end, :] = np.random.normal(mean, std, size=(self.height, col_end - col_start, self.frame_count))
    
        # Step 2: Per-cell replacement
        for region in self.cell_regions:
            yx_coords = region.coords
            cell_y, cell_x = region.centroid
    
            # Find closest cell profile by 2D centroid distance
            distances = [
                np.sqrt((p["x"] - cell_x)**2 + (p["y"] - cell_y)**2)
                for p in self.noise["cell_profiles"]
            ]
            closest_idx = np.argmin(distances)
            cell_mean = self.noise["cell_profiles"][closest_idx]["mean"]
            cell_std = self.noise["cell_profiles"][closest_idx]["std"]  # re-use bg std (or estimate cell std separately)
            # Replace all pixels in cell
            for (y, x) in yx_coords:
                self.movie[y, x, :] = np.random.normal(cell_mean, cell_std, size=self.frame_count)
    def _insert_molecule(self, x, y, t, profile):
        half = profile.shape[0] // 2
        x1, x2 = x - half, x + half + 1
        y1, y2 = y - half, y + half + 1

        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
            return

        self.movie[y1:y2, x1:x2, t] += profile
    

    def simulate(self, noise_scale=1.0, molecule_scale=1.0):
        """
        Assigns trajectories to cells and overlays the molecule profiles.
        """
        self.movie *= noise_scale
        for region in self.cell_regions:
            mask_coords = region.coords
            if len(mask_coords) == 0:
                continue

            sampled_trajs = random.choices(self.trajectory_groups, k=self.density_per_cell)

            for traj in sampled_trajs:
                anchor_y, anchor_x = random.choice(mask_coords)

                offset_x = anchor_x - int(np.round((traj['x'].mean())))
                offset_y = anchor_y - int(np.round((traj['y'].mean())))

                max_start_frame = self.frame_count - int(traj['frame'].max()) - 1
                if max_start_frame < 0:
                    continue

                start_offset = np.random.randint(0, max_start_frame + 1)
                adjusted_traj = []  # List of rows
                for _, row in traj.iterrows():
                    xf = row['x'] + offset_x
                    yf = row['y'] + offset_y
                    t = int(row['frame']) + start_offset
                    if t >= self.frame_count:
                        continue
                
                    # Subpixel binning
                    (bin_y, bin_x), (flip_y, flip_x) = get_subpixel_bin_and_flips(xf, yf, bin_size=self.bin_size)
                
                    try:
                        profile = random.choice(self.molecule_profiles[(bin_y, bin_x)])
                    except KeyError:
                        continue  # skip if bin is empty
                
                    profile = flip_profile(profile, flip_y, flip_x)
                    profile = np.clip(profile - np.median(profile), 0, None)
                    profile *= molecule_scale
                
                    x = int(np.round(xf))
                    y = int(np.round(yf))
                    self._insert_molecule(x, y, t, profile)
                    adjusted_traj.append((xf, yf, t))

                # Store the full adjusted trajectory
                if adjusted_traj:
                    df_traj = pd.DataFrame(adjusted_traj, columns=['x', 'y', 't'])
                    self.inserted_trajectories.append(df_traj)
        return

    def save_movie(self, filename='simulated_movie.tif'):
        """
        Saves the 3D movie as an uncompressed TIFF stack.

        Args:
            filename (str): Output filename.
        """
        # Reorder to (frames, height, width) for TIFF convention
        movie_stack = np.moveaxis(self.movie.astype(np.uint16), -1, 0)
        tifffile.imwrite(filename, movie_stack, photometric='minisblack')
    def save_track(self, filename = 'groundtruth_tracks.csv'):
        # Save trajectory
        all_trajs = pd.concat(
            [df.assign(traj_id=i) for i, df in enumerate(self.inserted_trajectories)],
            ignore_index=True
        )
        all_trajs.to_csv(filename, index=False)



tracks = strobe_multistate(
        100000,   # 10000 trajectories
        [0.5, 2 ,10.0],     # diffusion coefficient, microns squared per sec
        [0.3, 0.5, 0.2],     # state occupancies
        motion="brownian",
        geometry="sphere",
        radius=5.0,
        dz=0.7,
        frame_interval=0.01,
        loc_error=0.035,
        track_len=100,
        bleach_prob=0.05)
from columnwise_noise import analyze_binned_noise_stats
pixel_size = 0.108
tracks['x'] = tracks['x'] / pixel_size
tracks['y'] = tracks['y'] / pixel_size
# movie_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/v520_Halo_H2B/Dstorm'
# mask_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04202025/v520_Halo_H2B/single_mov_tracking/masks'
# movies = [tifffile.imread(file)[-10:,:,:] for file in glob.glob(movie_folder + '/*.tif')]
# masks =[io.imread(file) for file in glob.glob(mask_folder + '/*_seg.tif')]
# noises = analyze_binned_noise_stats(movies, masks)
import pickle
with open('H2B_calculated_bg_noise.pkl','rb') as fh:
    noises = pickle.load(fh)

def simulate_molecule_scale_series(
    output_dir,
    trajectories,
    cell_mask,
    molecule_profiles,
    noise,
    bin_size = 0.1,
    density_per_cell=50,
    max_frames=100,
    base_seed=818
):
    os.makedirs(output_dir, exist_ok=True)

    scale_values = np.linspace(0.6, 1.4, 5)
    for scale in scale_values:
        print(f"Simulating movie with molecule_scale = {scale:.2f}")

        simulator = Read_cell_ols(
            trajectories=trajectories,
            cell_mask=cell_mask,
            bind_profiles=molecule_profiles,
            noise=noise,
            density_per_cell=density_per_cell,
            max_frames=max_frames,
            bin_size = bin_size,
            random_seed=base_seed
        )

        simulator.simulate(molecule_scale=scale, noise_scale=1.0)
        simulator.save_movie(filename = os.path.join(output_dir, f"movie_mscale_{scale:.2f}.tif"))
        simulator.save_track(filename = os.path.join(output_dir, f"trajectories_mscale_{scale:.2f}.csv"))

# A = pd.read_pickle('/home/alineos1/Documents/codes/full_saved_regions.pkl')
# A = group_profiles_by_subpixel_bin_symmetry(A, fit_func=fit_gaussian_2d_55)
if __name__ == '__main__': 
    with open('Grouped_profiles.pkl','rb') as fh:
        grouped_profile = pickle.load(fh)
    A = grouped_profile[0]
    A_005 = grouped_profile[1]
    cell_mask1 = io.imread('/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_4D/Dstorm/masks/Mask_2_seg.tif')
    cell_mask2 = io.imread('/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_4D/Dstorm/masks/Mask_4_seg.tif')
    # cell_mask3 = io.imread('/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_4D/Dstorm/masks/Mask_6_seg.tif')
    cell_masks = [cell_mask1, cell_mask2] #, cell_mask3]
    for i, cell_mask in enumerate(cell_masks):
        output_dir = f'/home/alineos1/Documents/movie_simu/H2B/bin5/mask_{i}' 
        simulate_molecule_scale_series(output_dir=output_dir,
                                       trajectories=tracks, cell_mask=cell_mask, molecule_profiles=A,
                                       noise=noises, base_seed= 818*i)
    for i, cell_mask in enumerate(cell_masks):
        output_dir = f'/home/alineos1/Documents/movie_simu/H2B/bin10/mask_{i}' 
        simulate_molecule_scale_series(output_dir=output_dir,
                                       trajectories=tracks, cell_mask=cell_mask, molecule_profiles=A_005,
                                       noise=noises, bin_size=0.05, base_seed= 818*i)

