"""
utils.py

"""
import numpy as np 
import pandas as pd 

def sum_squared_jump(tracks, pixel_size_um=1.0):
    """
    For each trajectory in a dataset, calculate the sum
    of squared jumps in um. 

    args
    ----
        tracks          :   pandas.DataFrame, input trajectories
        pixel_size_um   :   float, size of pixels in um

    returns
    -------
        pandas.DataFrame, indexed by trajectory, with the columns
            "trajectory", "n_jumps", "sum_sq_jump"

    """
    # Work with a copy of the trajectories
    tracks = tracks.copy()

    # Exclude singlets
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] > 1]

    # Convert from pixels to um 
    tracks[['y', 'x']] = tracks[['y', 'x']] * pixel_size_um 

    # Sort
    tracks = tracks.sort_values(by=["trajectory", "frame"])
    n_tracks = tracks["trajectory"].nunique()

    # Calculate YX displacement vectors
    _T = np.asarray(tracks[['frame', 'trajectory', 'y', 'x',
        'track_length', 'track_length']])
    vecs = _T[1:,:] - _T[:-1,:]

    # Map trajectory indices back 
    vecs[:,5] = _T[:-1,1]
    vecs[:,4] = _T[:-1,4]

    # Only consider points originating from the same trajectory
    # and from subsequent frames
    take = np.logical_and(vecs[:,1] == 0.0, vecs[:,0] == 1.0)
    vecs = vecs[take, :]

    # Calculate squared 2D radial jumps
    vecs[:,1] = vecs[:,2]**2 + vecs[:,3]**2

    # Format as pandas.DataFrame for groupby
    M = vecs.shape[0]
    df = pd.DataFrame(index=np.arange(M), 
        columns=["sq_jump", "trajectory"])
    df["sq_jump"] = vecs[:,1]
    df["trajectory"] = vecs[:,5]
    df = df.join(
        df.groupby("trajectory").size().rename("n_jumps"),
        on="trajectory"
    )

    # Sum of squared displacements for each trajectory
    S = pd.DataFrame(index=np.arange(n_tracks), 
        columns=["sum_sq_jump", "trajectory", "n_jumps"])
    S["sum_sq_jump"] = np.asarray(df.groupby("trajectory")["sq_jump"].sum())
    S["trajectory"] = np.asarray(df.groupby("trajectory").apply(lambda i: i.name))
    S["n_jumps"] = np.asarray(df.groupby("trajectory").size())

    return S

def track_length(tracks):
    """
    Calculate the number of points in each of a set of trajectories.

    args
    ----
        tracks      :   pandas.DataFrame

    returns
    -------
        same dataframe with "track_length" column 

    """
    if "track_length" in tracks.columns:
        tracks = tracks.drop("track_length", axis=1)
    tracks = tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )
    return tracks 
