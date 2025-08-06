import os
import re
import argparse
import math

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
    naming_syntax = "_Iter_", #for RegEX
    show = True 
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
        
        if show:
            show_validated_nuclei(
                tifffile.imread(nucleus_frame), masks, stats,
                key=key,
            )

    return all_stats

def show_validated_nuclei(nuc_img, masks, stats, key,
                        n_cols=5,
                        padding=5,
                        dpi=200):
    """
    Tiles all validated nuclei for one Iter (key) into a high-res grid.
    """
    valid_labels = [s['nucleus_label'] for s in stats if s['validated']]
    if not valid_labels:
        print(f"[Iter {key}] No validated nuclei to show.")
        return

    n = len(valid_labels)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 3),
        squeeze=False,
        dpi=dpi
    )
    axes = axes.flatten()

    props = {p.label: p for p in regionprops(masks)}

    for idx, lab in enumerate(valid_labels):
        ax = axes[idx]
        p = props[lab]
        minr, minc, maxr, maxc = p.bbox
        minr, minc = max(minr - padding, 0), max(minc - padding, 0)
        maxr = min(maxr + padding, nuc_img.shape[0])
        maxc = min(maxc + padding, nuc_img.shape[1])

        crop = nuc_img[minr:maxr, minc:maxc]
        crop_mask = (masks[minr:maxr, minc:maxc] == lab)

        ax.imshow(crop, cmap='gray')
        overlay = label2rgb(
            crop_mask.astype(int),
            image=crop,
            bg_label=0,
            alpha=0.35
        )
        ax.imshow(overlay)
        ax.contour(crop_mask, colors='r', linewidths=1)
        ax.set_title(f"ID {lab}", fontsize=8)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f"Validated nuclei — Iter {key}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Mask nuclei & QC-plot validated cells"
    )
    p.add_argument('nuclei_folder', help="Folder of nucleus-channel TIFFs")
    p.add_argument('signal_folder', help="Folder of signal-channel TIFFs")
    p.add_argument('--diameter','-d', type=float, default=60,
                help="Cellpose nucleus diameter (px)")
    p.add_argument('--percentile', type=float, default=90,
                help="Percentile threshold for bright-area")
    p.add_argument('--max_area', type=int, default=300,
                help="Max bright-area (px²) for validation")
    p.add_argument('--intensity_factor', type=float, default=2.0,
                help="Fold-change over rest required")
    p.add_argument('--n_cols', type=int, default=5,
                help="Columns in QC grid")
    p.add_argument('--padding', type=int, default=5,
                help="Pixel padding around each nucleus crop")
    p.add_argument('--show', '-v', action='store_true',
                help="Display tiled QC for validated nuclei")

    args = p.parse_args()

    process_movie(
        args.nuclei_folder,
        args.signal_folder,
        show=args.show
    )
