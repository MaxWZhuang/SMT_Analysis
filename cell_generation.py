
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label, find_objects, sum as ndi_sum

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import sys
import os
import glob

import saspt, strobesim
from saspt import sample_detections, StateArray, RBME
from strobesim import strobe_multistate

class Cell:
    """
        Represents a cell for a SMT movie. 
    """
    in_cell_noise = (112, 14)
    
    def __init__(self, cell_radius, trajectory, molecules = "/Users/maxzhuang/lab/OLS/full_saved_regions.pkl"):
        """
        Creates a cell to impose onto simulated movie. 
        Because the returned cells are empty and will have trajectories, then backgrounds imposed on them, returned nparrays are square. 
        Args:
            cell_radius (float): Radius of cell in pixels
            trajectory (DataFrame): Trajectory DataFrame to superimpose trajectories onto the cells. \
                Trajectory length determines the amount of frames in each cell. 
        Returns: 
            cell (ndarray): Returns a 3D (x, y, t) array representing trajectories imposed onto cells. 
        """
        self.cell_radius = cell_radius
        self.trajectory = trajectory
        self.molecules = [region for region in pd.read_pickle(molecules) if region.shape == (7, 7)]
        cell = np.zeros((int(self.cell_radius*2), int(self.cell_radius*2), int(self.trajectory["frame"].max()) + 1))
        for index, row in self.trajectory.iterrows():
            cell[int(row.x), int(row.y), int(row.frame)] = -1
        self.cell = cell
    
    def impose_molecules(self): 
        """
        Updates cells to have molecules imposed onto trajectories. 
        """
        zeroed_molecules = []
        for molecule in self.molecules: 
            zeroed_molecules.append((np.clip(molecule - np.median(molecule), a_min = 0, a_max = molecule.max())).astype("uint16"))
        
        for i in range(len(zeroed_molecules[0]), len(self.cell) - len(zeroed_molecules[0])): 
            # Beginning molecule imposition at the edge of the collected molecule, and ending at the cell 
            for j in range(len(zeroed_molecules[0][0]), len(self.cell[i]) - len(zeroed_molecules[0])):
                for f in range(len(self.cell[i][j])):
                    if self.cell[i][j][f] < 0: 
                        imposed_molecule = zeroed_molecules[np.random.choice(len(zeroed_molecules))]
                        self.cell[i - (len(imposed_molecule)//2):i + (len(imposed_molecule)//2) + 1, 
                                j - (len(imposed_molecule[0])//2) : j + (len(imposed_molecule[0])//2) + 1, f] += imposed_molecule           
        self.cell += np.random.normal(Cell.in_cell_noise[0], Cell.in_cell_noise[1], size = self.cell.shape)
    
    #####################
    # PLOTTING METHODS #
    ####################
    def show_cell_frame(self, frame, title = "Simulated Molecules in Cell"): 
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#272b30'
        plt.rcParams['image.cmap'] = 'gray'

        fig, ax = plt.subplots(layout="constrained")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad = 0.05)

        color_map = ax.imshow(self.cell[:, :, frame])
        fig.colorbar(color_map, cax = cax)
        img = ax.imshow(self.cell[:, :, frame])
    
    def save_movie(self, title = "Simulated Molecules in Frame", file_name = "mols_in_cell"): 
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#272b30'
        plt.rcParams['image.cmap'] = 'gray'

        cell_color_ref = self.cell[:, :, 0]

        fig, ax = plt.subplots(layout="constrained")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=20)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad = 0.05)

        color_map = ax.imshow(cell_color_ref)
        fig.colorbar(color_map, cax = cax)


        def iterate_through_cell(frame):
            print(frame)
            img = ax.imshow(self.cell[:, :, frame])
            ax.set_title(f"{title} # {frame}", fontsize=20)
            return img

        anim = FuncAnimation(
            fig, 
            iterate_through_cell,
            frames = self.cell.shape[2],
            interval = 400,
            blit = False
        )

        anim.save(f"{file_name}.gif")
        
