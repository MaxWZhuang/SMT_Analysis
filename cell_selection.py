#!/usr/bin/env python3
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from cellpose import models
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.filters import gaussian
from skimage.color import label2rgb

def segment_nuclei(frame, model, diameter=None):
    """
    Run Cellpose on a single grayscale frame (using the Hoscht label) to get an integer label mask.
    """
    masks, flows, styles, diams = model.eval(
        tifffile.imread(frame), diameter=diameter, channels=[0,0]
    )
    return masks

def analyze_frame_validate(
    masks,
    signal_frame,
    percentile=90,
    max_area=300,
    intensity_factor=2.0
):
    """
    Segment nuclei and in each nucleus find the single brightest contiguous area.
    Then mark that nucleus as 'validated' only if:
    • area_of_brightest <= max_area
    • mean_intensity_of_brightest >= intensity_factor * mean_intensity_of_rest
    
    Returns:
    • masks (2D array),
    • list of dicts with per-nucleus stats including a 'validated' flag
    """
    # 1) segment nuclei
    signal_image = tifffile.imread(signal_frame)
    nuc_props = regionprops(masks, intensity_image=signal_image)
    results = []

    for prop in nuc_props:
        lab = prop.label #label of the RegionProperties object, which is a nucleus
        region_mask = (masks == lab) # only the arrays where the pixels belong to the nucleus
        intensities = signal_image[region_mask] #1d array of all pixel values inside the nucleus

        # 2) threshold at the given percentile
        thr = np.percentile(intensities, percentile) # change this to be something like

        # 3) mask of candidate bright pixels
        bright_mask = (signal_image >= thr) & region_mask

        # 4) label connected bright areas
        bright_labels = label(bright_mask)
        if bright_labels.max() == 0:
            # no candidate bright region
            results.append({
                'nucleus_label': lab,
                'nucleus_area': int(prop.area),
                'validated': False,
                'reason': 'no bright area',
            })
            continue

        # 5) pick the region with highest total intensity (mean*area)
        best_reg = None
        best_score = -np.inf
        for r in regionprops(bright_labels, intensity_image=signal_image):
            total_int = r.mean_intensity * r.area
            if total_int > best_score:
                best_score = total_int
                best_reg = r

        # compute rest-of-nucleus intensity
        best_mask = (bright_labels == best_reg.label)
        rest_mask = region_mask & ~best_mask
        mean_rest = float(signal_image[rest_mask].mean()) if rest_mask.any() else 0.0
        mean_bright = float(best_reg.mean_intensity)

        # validation criteria
        is_small      = best_reg.area <= max_area
        is_bright_enough = (mean_bright >= intensity_factor * mean_rest)

        results.append({
            'nucleus_label':        lab,
            'nucleus_area':         int(prop.area),
            'bright_area':          int(best_reg.area),
            'bright_centroid_y':    float(best_reg.centroid[0]),
            'bright_centroid_x':    float(best_reg.centroid[1]),
            'mean_bright_int':      mean_bright,
            'mean_rest_int':        mean_rest,
            'threshold':            float(thr),
            'validated':            bool(is_small and is_bright_enough),
            'reason':               ("small_and_bright" if (is_small and is_bright_enough)
                                    else ("too_large" if not is_small else "not_bright"))
        })

    return masks, results

def process_movie(
    nuclei_folder, 
    signal_folder,
    naming_syntax = "_Iter_",
    show = True #for RegEX
): 
    """
    Processes sequences of images (one for nucleus labeling, one for finding signal). It uses a specific naming quirk to perform regex and 
    produce accurate matching.

    Args:
        nuclei_folder (_type_): Folder where the nucleus images are stored
        signal_folder (_type_): Folder where the signal images are stored
        naming_syntax (_type_, optional): _description_. Defaults to "_Iter_" forRegEX (where the following 4 numbers are used to match).
    """
    pattern = re.compile(rf"{naming_syntax}(\d+)")
    
    nuclei_files = {}
    for nuclei_file_name in os.listdir(nuclei_folder):
        full_pathname = os.path.join(nuclei_folder, nuclei_file_name)
        if not os.path.is_file(full_pathname):
            continue
        match = pattern.search(nuclei_file_name)
        if match: #if key exists
            nuclei_files[match.group(1)] = full_pathname
    
    signal_files = {}
    for signal_file_name in os.listdir(signal_folder):
        full_pathname = os.path.join(signal_folder, signal_file_name)
        if not os.path.is_file(full_pathname):
            continue
        match = pattern.search(signal_file_name)
        if match: #if key exists
            signal_files[match.group(1)] = full_pathname
            
    model = models.CellposeModel(pretrained_model="nuclei")
    
    if show: 
        fig, ax = plt.subplots(figsize = (16, 16), dpi = 200)
    
    all_stats = []
    for key in sorted(nuclei_files.keys()):
        if (key not in signal_files.keys()):
            continue
        nucleus_frame = nuclei_files.get(key)
        signal_frame = signal_files.get(key)
        
        masks, stats = analyze_frame_validate(
            segment_nuclei(nucleus_frame, model), 
            signal_frame,
            # optionally, you may also tweak thresholds here
        )
        all_stats.append(stats)

    return all_stats
    

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Mask nuclei & find brightest spots per nucleus"
    )
    p.add_argument('movie', help="4D TIFF (time,channel,y,x)")
    p.add_argument(
        '--mask_ch','-m',
        type=int, default=0,
        help="channel index for nuclei masking"
    )
    p.add_argument(
        '--signal_ch','-s',
        type=int, default=1,
        help="channel index for spot detection"
    )
    p.add_argument(
        '--diameter','-d',
        type=float, default=60,
        help="approximate nucleus diameter (px)"
    )
    p.add_argument(
        '--min_dist','-p',
        type=int, default=5,
        help="min distance between peaks (px)"
    )
    p.add_argument(
        '--show','-v',
        action='store_true',
        help="display overlays for each frame"
    )
    args = p.parse_args()

    stats = process_movie(
        args.movie,
        args.mask_ch,
        args.signal_ch,
        args.diameter,
        args.min_dist,
        args.show
    )

    # Example: print out summary for frame 0
    import pprint
    print("Frame 0 stats:")
    pprint.pprint(stats[0])
