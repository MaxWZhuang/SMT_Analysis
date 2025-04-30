#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:28:29 2025

@author: alineos1
"""

import numpy as np
from scipy.stats import poisson
from joblib import Parallel, delayed
import multiprocessing

class OLSMicroscopeSimulator:
    def __init__(
        self,
        cam_shape=(256, 256),
        pixel_size_um=0.108,
        na=1.2,
        emission_wavelength_um=0.68,
        frame_rate_hz=100,
        line_exposure_us=500,
        mean_photons=90,
        read_noise_std=3,
        gain=4.3,
        z_extent_um=6.3,
        z_planes=21,
        psf_size_px=15,
        n_jobs=None,
        dt_sim_us=100  # fine timestep in microseconds
    ):
        self.cam_shape = cam_shape
        self.pixel_size_um = pixel_size_um
        self.na = na
        self.emission_wavelength_um = emission_wavelength_um
        self.frame_rate_hz = frame_rate_hz
        self.line_exposure_us = line_exposure_us
        self.mean_photons = mean_photons
        self.read_noise_std = read_noise_std
        self.gain = gain
        self.z_extent_um = z_extent_um
        self.z_planes = z_planes
        self.psf_size_px = psf_size_px
        
        self.frame_time_s = 1.0 / frame_rate_hz
        self.line_time_s = self.frame_time_s / cam_shape[0]

        self.psf_stack = self._generate_psf_stack()
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        
        self.dt_sim_us = dt_sim_us
        self.dt_sim_s = dt_sim_us * 1e-6

    def _generate_psf_stack(self):
        """Simple 3D Gaussian PSF stack."""
        size = self.psf_size_px
        sigma_xy = 1.2
        sigma_z = 2.5
        psf_stack = []
        center = size // 2
        for z in np.linspace(-self.z_planes/2, self.z_planes/2, self.z_planes):
            z_factor = z / (self.z_planes / 2)
            psf = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    r2 = (i - center)**2 + (j - center)**2
                    psf[i, j] = np.exp(-r2 / (2 * sigma_xy**2)) * np.exp(-z_factor**2 / (2 * sigma_z**2))
            psf /= np.sum(psf)
            psf_stack.append(psf)
        return np.stack(psf_stack, axis=0)

    def _current_shutter_row(self, t_frame_us):
        """Given time within a frame (us), compute current active shutter row."""
        line_time_us = self.line_time_s * 1e6
        return t_frame_us / line_time_us

    def simulate_diffusion_track(self, x0_um, y0_um, z0_um, D_um2_s, n_frames):
        """
        Simulate fine-step diffusion for one molecule across multiple frames.
        Returns fine time-sampled (x,y,z,t) array.
        """
        total_time_us = n_frames * self.frame_time_s * 1e6
        n_steps = int(total_time_us / self.dt_sim_us)
        
        positions = np.zeros((n_steps, 4))  # x, y, z, t
        positions[0, :3] = (x0_um, y0_um, z0_um)

        std_dev = np.sqrt(2 * D_um2_s * self.dt_sim_s)

        for i in range(1, n_steps):
            steps = np.random.normal(0, std_dev, 3)
            positions[i, 0:3] = positions[i-1, 0:3] + steps
            positions[i, 3] = i * self.dt_sim_us

        return positions

    def _simulate_frame(self, fine_positions, frame_idx):
        """Render a frame by checking which fine-step points are illuminated."""
        frame = np.zeros(self.cam_shape, dtype=np.float32)

        frame_start_us = frame_idx * self.frame_time_s * 1e6
        frame_end_us = (frame_idx + 1) * self.frame_time_s * 1e6

        for (x_um, y_um, z_um, t_us) in fine_positions:
            if not (frame_start_us <= t_us < frame_end_us):
                continue

            row_y = self._current_shutter_row(t_us - frame_start_us)
            y_px = y_um / self.pixel_size_um

            if abs(y_px - row_y) <= (self.line_exposure_us / (2 * self.line_time_s * 1e6)):
                psf_info = self.simulate_psf(x_um, y_um, z_um, frame_idx)
                if psf_info is not None:
                    psf_patch, (i0, j0) = psf_info
                    half_size = self.psf_size_px // 2
                    i_start, j_start = i0 - half_size, j0 - half_size
                    i_end, j_end = i_start + self.psf_size_px, j_start + self.psf_size_px

                    if (i_start < 0 or j_start < 0 or
                        i_end > self.cam_shape[0] or j_end > self.cam_shape[1]):
                        continue

                    frame[i_start:i_end, j_start:j_end] += psf_patch * self.mean_photons

        frame = poisson.rvs(frame)
        noise = np.random.normal(0, self.read_noise_std, frame.shape)
        frame = (frame + noise) * self.gain
        return np.clip(frame, 0, 65535).astype(np.uint16)

    def simulate_movie_from_diffusion(self, initial_positions, D_um2_s, n_frames):
        """
        Simulate movie from multiple molecules diffusing.

        Args:
            initial_positions: list of (x0_um, y0_um, z0_um)
            D_um2_s: diffusion coefficient
            n_frames: number of frames

        Returns:
            movie: (n_frames, H, W) numpy array
        """
        # Fine simulation for each particle
        all_fine_positions = []
        for x0, y0, z0 in initial_positions:
            pos = self.simulate_diffusion_track(x0, y0, z0, D_um2_s, n_frames)
            all_fine_positions.append(pos)

        all_fine_positions = np.vstack(all_fine_positions)

        # Parallel rendering
        movie = Parallel(n_jobs=self.n_jobs)(
            delayed(self._simulate_frame)(all_fine_positions, f)
            for f in range(n_frames)
        )

        return np.stack(movie, axis=0)

    def simulate_psf(self, x_um, y_um, z_um, frame_idx):
        """Project molecule at (x, y, z) to PSF centered at nearest pixel."""
        x_px = x_um / self.pixel_size_um
        y_px = y_um / self.pixel_size_um
        z_idx = int((z_um / self.z_extent_um) * self.z_planes)
        z_idx = np.clip(z_idx, 0, self.z_planes - 1)

        psf = self.psf_stack[z_idx]
        j0 = int(np.round(x_px))
        i0 = int(np.round(y_px))

        return psf, (i0, j0)
