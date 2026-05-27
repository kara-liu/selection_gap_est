import numpy as np
from numpy.random import SeedSequence, default_rng, PCG64
from scipy.special import expit
from scipy import optimize
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from typing import Sequence, Optional, List
from utils import MethodConfig
import pandas as pd



def _bootstrap_iter_exact(P, Q_moments, rng):
    """
    One bootstrap iteration using exact resampling.
    """
    P_bs = rng.choice(P, size=P.shape[0], replace=True)

    def moment_residual(theta):
        pS = expit(P_bs @ theta)
        P_hat = np.average(P_bs, weights=1.0 / pS, axis=0)
        return P_hat - Q_moments

    sol = optimize.root(moment_residual, x0=np.zeros(P.shape[1]))
    return sol.x


def _blocked_gram(P, factor, block):
    d = P.shape[1]
    G = np.zeros((d, d))
    for i in range(0, len(P), block):
        Pi = P[i:i+block]
        fi = factor[i:i+block]
        G += Pi.T @ (Pi * fi[:, None])
    return G


def _blocked_Pt_vec(P, v, block):
    out = np.zeros(P.shape[1])
    for i in range(0, len(P), block):
        out += P[i:i+block].T @ v[i:i+block]
    return out


def _residual_and_jac(P, Q, theta, counts, block):
    logits = np.clip(P @ theta, -20.0, 20.0)
    pS = expit(logits)
    inv_pS = 1.0 / (pS + 1e-12)

    w = counts * inv_pS
    s = w.sum() + 1e-12

    P_sum = _blocked_Pt_vec(P, w, block) / s
    r = P_sum - Q

    factor = -counts * (1.0 - pS) * inv_pS
    ds = _blocked_Pt_vec(P, factor, block)
    PTG = _blocked_gram(P, factor, block)

    J = (PTG / s) - np.outer(P_sum, ds) / s
    return r, J


def _solve_theta(P, Q, theta0, counts, block=65536, maxiter=120):
    def fun(theta):
        r, _ = _residual_and_jac(P, Q, theta, counts, block=block)
        return r
    def jac(theta):
        _, J = _residual_and_jac(P, Q, theta, counts, block=block)
        return J

    res = optimize.least_squares(
        fun, theta0, jac=jac, method="trf",
        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=maxiter
    )
    return res.x

def _bootstrap_chunk(P, Q, rng, iters, theta0, block=65536):
    """
    Process `iters` bootstraps sequentially with warm starts inside one worker.
    Poisson(1) bootstrap by default (≈ multinomial for large n).
    """
    n, d = P.shape
    thetas = np.empty((iters, d), dtype=P.dtype)
    th = theta0.copy()

    for t in range(iters):
        counts = rng.poisson(1.0, size=n)

        # Ensure at least one nonzero (extremely unlikely to be all-zero with Poisson)
        if counts.sum() == 0:
            counts[rng.integers(0, n)] = 1

        thetas[t] = _solve_theta(P, Q, th, counts, block=block)
    return thetas


def bootstrap_moment_matching(
    P_candidates, Q_moments,
    mm_type='fast',
    bootstrap_iters=1000,
    n_jobs=14,                 
    seed=42,
    chunk_size=8,            
    block=65536,             # rows per blocked BLAS; tune to cache
):
    """
    Bootstrap moment-matching heuristic. 
    """
    assert mm_type in ['fast','exact']
    if mm_type == 'exact':
        ss = SeedSequence(seed)
        child_seeds = [ss.generate_state(1)[0] for ss in ss.spawn(bootstrap_iters)]
        res = Parallel(n_jobs=n_jobs)(
                    delayed(_bootstrap_iter_exact)(P_candidates, Q_moments, np.random.default_rng(PCG64(seed)))
                    for seed in child_seeds
        )
        return np.vstack(res)
    elif mm_type == 'fast':
        P = np.ascontiguousarray(P_candidates, dtype=np.float64)
        Q = np.asarray(Q_moments, dtype=np.float64)
        d = P.shape[1]

        # Global warm-start (solve once on counts=1)
        theta0 = _solve_theta(P, Q, theta0=np.zeros(d), counts=np.ones(P.shape[0]), block=block, maxiter=80)

        # RNGs per chunk
        ss = SeedSequence(seed)
        n_chunks = int(np.ceil(bootstrap_iters / chunk_size))
        rngs = [default_rng(PCG64(s)) for s in ss.spawn(n_chunks)]

        with threadpool_limits(limits=1, user_api="blas"):
            chunks = Parallel(n_jobs=n_jobs, prefer='threads', batch_size=1)(
                delayed(_bootstrap_chunk)(
                    P, Q, rng,
                    iters=(chunk_size if i < n_chunks - 1 else (bootstrap_iters - (n_chunks - 1)*chunk_size)),
                    theta0=theta0,
                    block=block,
                )
                for i, rng in enumerate(rngs)
            )

        return np.vstack(chunks)


def _prefilter_candidates_by_cohen(
    P_candidates,
    P_data,
    Q_mu,
    Q_var,
    Q_N,
    threshold,
):
    """
    Reduce candidate variable set by filtering using Cohen's d statistics.
    """
    P_N = len(P_data)
    P_var = np.var(P_data, axis=0, ddof=1)
    P_mu = np.mean(P_data, axis=0)
    pooled_std = np.sqrt(((P_N - 1) * P_var + (Q_N - 1) * Q_var) / (P_N + Q_N - 2))

    cohen_d = np.abs(P_mu - Q_mu) / pooled_std
    keep = (cohen_d > threshold).squeeze()
    return list(np.asarray(P_candidates)[keep])
 

def run_heuristic_mm_ustar(
    P_df: pd.DataFrame,
    Utilde_cols: Sequence[str],
    all_Ustar_candidate_cols: Sequence[str],
    Q_mu: np.ndarray,
    config: MethodConfig,
    Q_var: Optional[np.ndarray] = None,
    Q_N: Optional[int] = None,
) -> List[str]:
    """
    Moment-matching heuristic to nominate unobserved selection variables U*.
    Parameters
    ----------
    utilde_cols : sequence of str
        Observed selection variables (U~).
    all_ustar_candidate_cols : sequence of str
        Candidate unobserved selection variables.
    Q_mu : np.ndarray, optional
        Used as the moment for the moment-matching heuristic. 
    Q_var : np.ndarray, optional
        Variance of Q over [Utilde | candidate] columns.
        Required only if Cohen prefiltering is enabled.
    Q_N : int, optional
        Sample size of Q.
        Required only if Cohen prefiltering is enabled.
    config : MethodConfig
        Configuration dictionary

    Returns
    -------
    nominated_ustar_cols : list[str]
        Candidate columns nominated as U*.
    """
    assert len(np.intersect1d(Utilde_cols, all_Ustar_candidate_cols)) == 0, "Utilde and candidate sets must be disjoint."

    # Filter candidate variables if in high-dimensional setting
    if config.heuristic_high_dim is not None and len(all_Ustar_candidate_cols) > config.heuristic_high_dim:
        if Q_var is None or Q_N is None:
            raise ValueError
    
        all_Ustar_candidate_cols = _prefilter_candidates_by_cohen(
            all_Ustar_candidate_cols,
            P_data=P_df[list(all_Ustar_candidate_cols)],
            Q_mu=Q_mu[list(all_Ustar_candidate_cols)],
            Q_N=Q_N, 
            Q_var=Q_var[list(all_Ustar_candidate_cols)],
            threshold=0.05,
        )


    bootstrap_thetas = bootstrap_moment_matching(
        P_df[list(Utilde_cols) + list(all_Ustar_candidate_cols)].to_numpy(),
        Q_mu[list(Utilde_cols) + list(all_Ustar_candidate_cols)].to_numpy().squeeze() ,
        bootstrap_iters=config.heuristic_bootstrap,
        mm_type=config.heuristic_type,
        seed=42,
        n_jobs=-1 # parallelize bootstrap iterations across CPU cores; adjust as needed
    )
    
    candidate_thetas = bootstrap_thetas[:,len(Utilde_cols):]
    cis = np.quantile(candidate_thetas, q=[config.heuristic_alpha, 1-config.heuristic_alpha], axis=0)
    contains_zero = (cis[0, :] <= 0) & (cis[1, :] >= 0)
    nominated_ustar = list(np.array(all_Ustar_candidate_cols)[~contains_zero])

    avg_thetas = np.mean(bootstrap_thetas, axis=0) # we treat the weight vector as the average theta
    weight = 1 / expit(P_df[list(Utilde_cols) + list(all_Ustar_candidate_cols)] @ avg_thetas)
    
    return nominated_ustar, weight

