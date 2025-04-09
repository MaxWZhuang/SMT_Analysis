#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:18:26 2025

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

class Read_cell_ols:
    def __init__(self, trajectories, cell_mask, molecule_profiles, noise, density_per_cell=10, max_frames=10, random_seed=None):
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
        self.molecule_profiles = [m for m in molecule_profiles if m.shape == (7, 7)]
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

                offset_x = anchor_x - int(traj['x'].mean())
                offset_y = anchor_y - int(traj['y'].mean())

                max_start_frame = self.frame_count - int(traj['frame'].max()) - 1
                if max_start_frame < 0:
                    continue

                start_offset = np.random.randint(0, max_start_frame + 1)
                adjusted_traj = []  # List of rows
                for _, row in traj.iterrows():
                    x = int(row['x'] + offset_x)
                    y = int(row['y'] + offset_y)
                    t = int(row['frame']) + start_offset
                    if t >= self.frame_count:
                        continue
                
                    profile = random.choice(self.molecule_profiles)
                    profile = np.clip(profile - np.median(profile), 0, None)
                    profile *= molecule_scale 
                    self._insert_molecule(x, y, t, profile)
                    adjusted_traj.append((x, y, t))
                
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

A = pd.read_pickle('/home/alineos1/Documents/codes/full_saved_regions.pkl')
cell_mask = io.imread('/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_4D/Dstorm/masks/Mask_2_seg.tif')

tracks = strobe_multistate(
        100000,   # 10000 trajectories
        [0.1, 3.0, 8.0],     # diffusion coefficient, microns squared per sec
        [0.3, 0.5, 0.2],     # state occupancies
        motion="brownian",
        geometry="sphere",
        radius=5.0,
        dz=0.7,
        frame_interval=0.01,
        loc_error=0.035,
        track_len=100,
        bleach_prob=0.1)
from columnwise_noise import analyze_binned_noise_stats

# movie_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_12F/Dstorm'
# mask_folder = '/media/alineos1/42368BE57732CF06/Swept_Hilo/04022025/Halo_REST_12F/Dstorm/masks'
# movies = [tifffile.imread(file)[-10:,:,:] for file in glob.glob(movie_folder + '/*.tif')]
# masks =[io.imread(file) for file in glob.glob(mask_folder + '/*_seg.tif')]
# noises = analyze_binned_noise_stats(movies, masks)
import pickle
with open('calculated_bg_noise.pkl','rb') as fh:
    noises = pickle.load(fh)

def simulate_molecule_scale_series(
    output_dir,
    trajectories,
    cell_mask,
    molecule_profiles,
    noise,
    density_per_cell=100,
    max_frames=200,
    base_seed=818
):
    os.makedirs(output_dir, exist_ok=True)

    scale_values = np.linspace(0.6, 1.5, 10)
    for scale in scale_values:
        print(f"Simulating movie with molecule_scale = {scale:.2f}")

        simulator = Read_cell_ols(
            trajectories=trajectories,
            cell_mask=cell_mask,
            molecule_profiles=molecule_profiles,
            noise=noise,
            density_per_cell=density_per_cell,
            max_frames=max_frames,
            random_seed=base_seed
        )

        simulator.simulate(molecule_scale=scale, noise_scale=1.0)
        simulator.save_movie(filename = os.path.join(output_dir, f"movie_mscale_{scale:.2f}.tif"))
        simulator.save_track(filename = os.path.join(output_dir, f"trajectories_mscale_{scale:.2f}.csv"))


simulate_molecule_scale_series(output_dir='/home/alineos1/Documents/movie_simu',
                               trajectories=tracks, cell_mask=cell_mask, molecule_profiles=A,
                               noise=noises)

