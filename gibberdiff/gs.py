#!/usr/bin/env python
"""
gs.py -- core Gibbs sampler

"""
from tqdm import tqdm 
import numpy as np
from scipy.special import ndtr

from .utils import sum_squared_jump, track_length
from .defoc import f_remain

# Keyed from n_states to the initial guess
INIT_DIFF_COEFS = {
   1: np.array([1.0]),
   2: np.array([0.1, 2.0]),
   3: np.array([0.01, 1.0, 5.0]),
   4: np.array([0.01, 0.5, 2.0, 8.0]),
   5: np.array([0.01, 0.5, 1.0, 3.0, 10.0]) 
}

def gibberdiff(tracks, n_states=2, frame_interval=0.01, loc_error=0.035, 
    dz=None, pixel_size_um=0.16, n_iter=1000, burnin=30, 
    metropolis_steps=5, metropolis_scale=0.05, bounds=(0, 100.0),
    pseudocount_frac=0.001, ret_samples=False):
    """
    Use a Gibbs sampling routine to estimate the parameters for
    a finite-state mixture of regular Brownian motions.

    important
    ---------
        1. This function does not deal with identifiability issues.
        2. The localization error is assumed to be known in advance.

    args
    ----
        data            :   pandas.DataFrame
        n_states        :   int
        frame_interval  :   float, seconds
        loc_error       :   float, localization error in um
        dz              :   float, focal depth in um
        pixel_size_um   :   float, size of pixels in um
        n_iter          :   int, the total number of iterations to run
        burnin          :   int, the number of iterations to do before 
                            recording anything
        metropolis_steps:   int, the number of Metropolis-Hastings steps
                            to use to update the diffusion coefficients
                            at each step
        metropolis_scale:   float, the standard deviations of the proposal
                            distributions for the Metropolis-Hastings 
                            step
        bounds          :   2-tuple of float, the lower and upper bounds
                            on the diffusion coefficient
        pseudocount_frac:   float, the number of pseudocounts in the prior
                            over state occupations, expressed as a fraction
                            of the total number of trajectories
        ret_samples     :   bool, also return the raw samples from the 
                            posterior distribution

    returns
    -------
        (
            1D ndarray of shape (n_states,), the posterior mean over
                state occupations;
            1D ndarray of shape (n_states,), the posterior mean over
                diffusion coefficients
        )

    """
    le2 = loc_error ** 2
    Dmin, Dmax = bounds 

    # Calculate the squared displacements corresponding to each 
    # trajectory
    S = sum_squared_jump(tracks, pixel_size_um=pixel_size_um)
    n_tracks = len(S)
    data = np.asarray(S[["sum_sq_jump", "n_jumps"]])

    # Model prior
    alpha = np.ones(n_states) * n_tracks * pseudocount_frac / n_states 

    # Log likelihood function of a diffusion coefficient, 
    # given some trajectories
    def log_likelihood(data, D):
        var2 = 4 * (D * frame_interval + le2)
        return -data[:,0] / var2 - data[:,1] * np.log(var2)

    # Initial parameter guesses
    occs = np.ones(n_states, dtype=np.float64) / n_states 
    if n_states in INIT_DIFF_COEFS.keys():
        diff_coefs = INIT_DIFF_COEFS[n_states]
    else:
        diff_coefs = np.sort(np.random.uniform(Dmin, Dmax, size=n_states))

    # Posterior mean over model parameters
    if not ret_samples:
        post_occs = np.zeros(n_states, dtype=np.float64)
        post_diff_coefs = np.zeros(n_states, dtype=np.float64)
    else:
        post_occs = np.zeros((n_iter, n_states), dtype=np.float64)
        post_diff_coefs = np.zeros((n_iter, n_states), dtype=np.float64)

    # Buffers
    L = np.zeros((n_tracks, n_states), dtype=np.float64)
    unassigned = np.zeros(n_tracks, dtype=np.bool)
    Z = np.zeros(n_tracks, dtype=np.uint16)
    cdf = np.zeros(n_tracks, dtype=np.float64)
    states = np.arange(n_states+1).astype(np.int64)

    for iter_idx in tqdm(range(n_iter)):

        # Evaluate the likelihood function on each pairwise
        # combination of trajectory and state, given the current
        # model parameters
        for j, D in enumerate(diff_coefs):
            L[:,j] = log_likelihood(data, D)

        # Normalize
        L = (L.T - L.max(axis=1)).T 
        L = np.exp(L)
        L = L * occs
        L = (L.T / L.sum(axis=1)).T 

        # For each trajectory, sample a new state
        u = np.random.random(size=n_tracks)
        unassigned[:] = True 
        cdf[:] = 0
        for j in range(n_states):
            cdf += L[:,j]
            match = np.logical_and(u<=cdf, unassigned)
            Z[match] = j 
            unassigned[match] = False 
        Z[unassigned] = n_states - 1 

        # Count the number of jumps in each state
        n = np.histogram(Z, bins=states, weights=data[:,1])[0].astype(np.float64)

        # Correct for defocalization
        if (not dz is None):
            corr = np.ones(n_states)
            for j, D in enumerate(diff_coefs):
                corr[j] = f_remain(D, 1, frame_interval, dz)[0]
            n /= corr 

        # Sample a new state occupation vector
        occs = np.random.dirichlet(n + alpha)

        # Sample new diffusion coefficients for each state
        for j, D in enumerate(diff_coefs):
            diff_coefs[j] = metropolis_step(
                D,
                data[Z==j, :],
                log_likelihood,
                metropolis_scale,
                bounds=bounds,
                n_steps=metropolis_steps
            )

        # Record 
        if not ret_samples:
            if iter_idx >= burnin:
                post_occs += occs 
                post_diff_coefs += diff_coefs 
        else:
            post_occs[iter_idx, :] = occs 
            post_diff_coefs[iter_idx, :] = diff_coefs 

    if not ret_samples:

        # Normalize
        post_occs /= (n_iter - burnin)
        post_diff_coefs /= (n_iter - burnin)

        # Order by ascending diffusion coefficient
        order = np.argsort(post_diff_coefs)
        post_diff_coefs = post_diff_coefs[order]
        post_occs = post_occs[order]

    return post_occs, post_diff_coefs

def propose_parameters(curr, scales, bounds=None):
    """
    Propose a new set of parameters for a Metropolis-Hastings
    step.

    args
    ----
        curr        :   1D ndarray of shape (M,), the current
                        set of parameters
        scales      :   1D ndarray of shape (M,), the standard
                        deviation for the step along each dimension
        bounds      :   2-tuple of 1D ndarray of shape (M,), the 
                        lower and upper bound for each parameter
    
    returns
    -------
        (
            1D ndarray of shape (M,), the new parameter proposal;
            1D ndarray of shape (M,), the bias terms for computing
                the Metropolis acceptance probability
        )

    """
    if isinstance(curr, float):

        # Make the proposal 
        prop = curr + np.random.normal(scale=scales)
        if not bounds is None:
            while (prop < bounds[0]) or (prop > bounds[1]):
                prop = curr + np.random.normal(scale=scales)

        # Calculate the bias term
        if not bounds is None:
            bias = (
                ndtr((bounds[1] - curr) / scales) - \
                ndtr((bounds[0] - curr) / scales)
            ) / (
                ndtr((bounds[1] - prop) / scales) - \
                ndtr((bounds[0] - prop) / scales)               
            )
        else:
            bias = 1.0

    elif isinstance(curr, np.ndarray):

        # Make the proposal
        prop = curr + np.random.normal(scale=scales)
        if not bounds is None:
            outside = np.logical_or(prop < bounds[0], prop > bounds[1])
            while outside.any():
                prop[outside] = curr[outside] + np.random.normal(scale=scales[outside])
                outside = np.logical_or(prop < bounds[0], prop > bounds[1])

        # Calculate the bias term 
        if not bounds is None:
            bias = ((
                ndtr((bounds[1] - curr) / scales) - \
                ndtr((bounds[0] - curr) / scales)
            ) / (
                ndtr((bounds[1]-prop) / scales) - \
                ndtr((bounds[0]-prop) / scales)
            )).prod()
        else:
            bias = 1.0

    return prop, bias 

def metropolis_step(curr, data, log_likelihood, scales, bounds=None,
    n_steps=1, **kwargs):
    """
    Given a current parameter estimate and a likelihood model,
    produce a new parameter estimate using a Metropolis-Hastings
    step. 

    args
    ----
        curr            :   float or 1D ndarray of shape (n,), the current
                            parameter estimate
        data            :   variable, the data to use for evaluating
                            the likelihood function
        log_likelihood  :   function with signature (data, curr, **kwargs),
                            the log likelihood function
        scales          :   float or 1D ndarray of shape (n,), the standard
                            deviations for the nudges to the parameters
        bounds          :   2-tuple of float or 1D ndarray of shape (n,),
                            parameter bounds 
        n_steps         :   int, the number of Metropolis-Hastings steps
                            to make
        kwargs          :   to *log_likelihood*

    returns
    -------
        float or 1D ndarray of shape (n,), the new parameter sample

    """
    u = np.random.random(size=n_steps)

    # Evaluate the log likelihood of the current parameters
    curr_log_L = log_likelihood(data, curr, **kwargs).sum()
    if isinstance(curr_log_L, np.ndarray):
        curr_log_L = curr_log_L.sum()

    for j in range(n_steps):

        # Propose a new parameter set
        prop, bias = propose_parameters(curr, scales, bounds=bounds)

        # Evaluate the log likelihood of the new guess 
        prop_log_L = log_likelihood(data, prop, **kwargs)
        if isinstance(prop_log_L, np.ndarray):
            prop_log_L = prop_log_L.sum()

        # Calculate the likelihood ratio
        d = prop_log_L - curr_log_L 
        if d < 100:
            p = bias * np.exp(prop_log_L - curr_log_L)
        else:
            p = 1.0

        # Determine whether to accept the step
        if u[j] <= p:
            curr = prop 
            curr_log_L = prop_log_L

    return curr 
