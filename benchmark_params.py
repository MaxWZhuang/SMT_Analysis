
# Filepaths
import os
from glob import glob

# CLI
import argparse

# DataFrames
import pandas as pd

# Parallelization and progress
import dask
from dask.diagnostics import ProgressBar

# quot tools
from quot.subpixel import localize_frame
from quot.chunkFilter import ChunkFilter
from quot.findSpots import detect
from quot.read import read_config
from quot.core import retrack_file
import itertools
from quot_fast_track import retrack_files_threads
ACCEPTABLE_EXTS = [".nd2", ".tif", ".tiff"] 

def localize_frames(movie_path: str, 
                    n_threads: int=4, 
                    out_dir: str=None, 
                    **kwargs):
    """
    Run detection and subpixel localization by frame-wise parallelization.

    args
    ----
    movie_path  :   str, path to movie to track
    n_threads :   int, the number of threads to use
    out_dir     :   str, output directory. If None, save the output
                    to the same directory as the movie_path
    kwargs      :   tracking configuration, as read with 
                    quot.read.read_config

    output
    ------
    write       :   CSV file with subpixel-localized spots
    return      :   str, path to the CSV file, so we can track this later

    """
    # Check that the movie path is a file
    assert os.path.isfile(movie_path), f"{movie_path} is not a file!"

    # Create the output directory if it doesn't exist
    if out_dir is not None and not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    
    # If the config file does not contain a 
    # "filter" section, don't worry about it
    kwargs["filter"] = kwargs.get("filter", {})

    # Open an image file reader with some filtering settings, if provided
    with ChunkFilter(movie_path, **kwargs['filter']) as fh:
        # Driver function to run spot detection and subpixel localization
        @dask.delayed 
        def driver(frame_idx, frame):
            detections = detect(frame, **kwargs['detect'])
            return localize_frame(frame, 
                                  detections, 
                                  **kwargs['localize']).assign(frame=frame_idx)

        # Run the driver function on each frame lazily
        tasks = [driver(i, frame) for i, frame in enumerate(fh)]
        scheduler = "single-threaded" if n_threads == 1 else "processes"
        with ProgressBar():
            print(f"Detecting and localizing spots in {movie_path} with {n_threads} threads.")
            result = dask.compute(*tasks, 
                                  scheduler=scheduler, 
                                  num_workers=n_threads)

    locs = pd.concat(result, ignore_index=True, sort=False)

    # Adjust for start index
    locs['frame'] += kwargs['filter'].get('start', 0)

    # Save to a file
    if out_dir is not None:
        out_csv_path = os.path.join(out_dir, 
                                    f"{os.path.splitext(os.path.basename(movie_path))[-2]}_trajs_k_{config['detect']['k']}_w_{config['detect']['w']}_t_{config['detect']['t']}_ws_{config['localize']['window_size']}.csv")
    else:
        out_csv_path = os.path.splitext(movie_path)[0] + f"_trajs_k_{config['detect']['k']}_w_{config['detect']['w']}_t_{config['detect']['t']}_ws_{config['localize']['window_size']}.csv"
    
    locs.to_csv(out_csv_path, index=False)
    
    return out_csv_path


def quot_fast_track_config(target_path: str, 
                    config: dict, 
                    ext: str=".tif", 
                    n_threads: int=1, 
                    out_dir: str=None, 
                    contains: str="*"):
    # If passed an image file, track without checking for ext or contains
    if os.path.isfile(target_path) and os.path.splitext(target_path)[-1] in ACCEPTABLE_EXTS:
        files_to_track = [target_path]
    # Otherwise, track all files matching ext and contains in the directory
    elif os.path.isdir(target_path):
        files_to_track = glob(os.path.join(target_path, f"*{contains}*{ext}"))
    # Else raise an error
    else:
        raise RuntimeError(f"Not a file or directory: {target_path}")
    
    for file in files_to_track:
        localize_frames(file, 
                        n_threads=n_threads, 
                        out_dir=out_dir, 
                        **config)
    
    # Do tracking parallelized over the output files
    
   


if __name__ == "__main__":
    # List of parameters to experiment with
    k_values = [1.0]  # example values for k
    w_values = [9, 11]       # example values for w (window size)
    t_values = [10.0, 13.0, 18.0] # example values for t (threshold)
    # window_size_values = [9, 11] # example values for window_size
    config = read_config("settings.toml")

    # Iterate over all combinations of k, w, t, window_size
    for k, w, t in itertools.product(k_values, w_values, t_values):
        
        # Modify the configuration dictionary
        config['detect']['k'] = k
        config['detect']['w'] = w
        config['detect']['t'] = t
        config['localize']['window_size'] = w
        quot_fast_track_config('/home/alineos1/Documents/movie_simu/H2B/bin10/mask_0/', 
                        config, 
                        '.tif', 
                        27, 
                        '/home/alineos1/Documents/movie_simu/H2B/bin10/mask_0/' 
                        )
        quot_fast_track_config('/home/alineos1/Documents/movie_simu/H2B/bin5/mask_0/', 
                        config, 
                        '.tif', 
                        27, 
                        '/home/alineos1/Documents/movie_simu/H2B/bin5/mask_0/' 
                        )

    config = read_config("settings.toml")
    tracks_csv = glob('/home/alineos1/Documents/movie_simu/H2B/bin10/mask_0/movie*.csv')
    print(f"Tracking {len(tracks_csv)} files...")
    retrack_files_threads(tracks_csv, 
                          out_suffix=None, 
                          num_workers=27, 
                          **config)  
    tracks_csv = glob('/home/alineos1/Documents/movie_simu/H2B/bin5/mask_0/movie*.csv')
    print(f"Tracking {len(tracks_csv)} files...")
    retrack_files_threads(tracks_csv, 
                          out_suffix=None, 
                          num_workers=27, 
                          **config)  

