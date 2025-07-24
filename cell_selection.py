#!/usr/bin/env python3
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from cellpose import models
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.filters import gaussian
from skimage.color import label2rgb

def segment_nuclei(frame, model, diameter):
    """
    Run Cellpose on a single grayscale frame (using the Hoscht label) to get an integer label mask.
    """
    masks, flows, styles, diams = model.eval(
        frame, diameter=diameter, channels=[0,0]
    )
    return masks

def analyze_frame_validate(
    nuclei_frame,
    signal_frame,
    model,
    diameter,
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
    masks = segment_nuclei(nuclei_frame, model, diameter)
    nuc_props = regionprops(masks, intensity_image=signal_frame)
    results = []

    for prop in nuc_props:
        lab = prop.label #label of the RegionProperties object, which is a nucleus
        region_mask = (masks == lab) # only the arrays where the pixels belong to the nucleus
        intensities = signal_frame[region_mask] #1d array of all pixel values inside the nucleus

        # 2) threshold at the given percentile
        thr = np.percentile(intensities, percentile) # change this to be something like

        # 3) mask of candidate bright pixels
        bright_mask = (signal_frame >= thr) & region_mask

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
        for r in regionprops(bright_labels, intensity_image=signal_frame):
            total_int = r.mean_intensity * r.area
            if total_int > best_score:
                best_score = total_int
                best_reg = r

        # compute rest-of-nucleus intensity
        best_mask = (bright_labels == best_reg.label)
        rest_mask = region_mask & ~best_mask
        mean_rest = float(signal_frame[rest_mask].mean()) if rest_mask.any() else 0.0
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
    infile, mask_channel, signal_channel,
    diameter, min_distance, show
):
    # load the entire movie: assume shape (T, C, Y, X)
    movie = tifffile.imread(infile)
    assert movie.ndim == 4, "Expecting a 4D TIFF (time,channel,y,x)"

    # load Cellpose model once
    model = models.CellposeModel(pretrained_model='nuclei')

    all_stats = []
    for t in range(movie.shape[0]):
        nuc = movie[t, mask_channel]
        sig = movie[t, signal_channel]

        masks, stats = analyze_frame_validate(
            nuc, sig, model, diameter, min_distance
        )
        all_stats.append(stats)

        if show:
            # overlay nuclei on signal
            overlay = label2rgb(masks, image=sig, bg_label=0, alpha=0.4)
            fig, ax = plt.subplots(1,1, figsize=(6,6))
            ax.imshow(overlay, cmap='gray')
            # plot spot centers
            ys = [s['spot_y'] for s in stats]
            xs = [s['spot_x'] for s in stats]
            ax.scatter(xs, ys, c='r', s=30, label='brightest spot')
            ax.set_title(f"Frame {t}")
            ax.axis('off')
            plt.show()

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
