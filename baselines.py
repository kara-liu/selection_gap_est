import pandas as pd 
import numpy as np 
import weightipy as wp
import empirical_calibration as ec
from empirical_calibration.core import ConvergenceError
from utils import get_binary_bool

def _run_raking(P_df, Q_mu):
    binary_cols = get_binary_bool(P_df)
    P_df = P_df[P_df.columns[binary_cols]]
        
    marginal_dict = {}
    for col in P_df.columns:
        marginal_dict[col] = {}
        marginal_dict[col][1] = Q_mu[col].iloc[0]
        
    P_df = P_df.reset_index(drop=True)
    scheme = wp.scheme_from_dict(marginal_dict)
    df_weighted = wp.weight_dataframe(df=P_df, scheme=scheme)
    return df_weighted['weights'].to_numpy()

def _run_emp_calib(P_df, Q_mu): 
    try:
        return ec.maybe_exact_calibrate(P_df, Q_mu,
            objective=ec.Objective.ENTROPY,
            autoscale=True
            )[0]
    except ConvergenceError as e:
        print(e)
        print("Calib failed, setting uniform weights")
        return np.ones(len(P_df))
        
def _run_ent_bal(P_df, Q_mu, Q_var): 
        Q_stats = Q_mu.copy()
        for column in P_df.columns:
            Q_stats[column+'_var_moment'] = Q_mu[column].iloc[0]**2 + Q_var[column].iloc[0]
            P_df[column+'_var_moment'] = P_df[column]**2
        try:
            return ec.maybe_exact_calibrate(P_df, Q_stats,
                objective=ec.Objective.ENTROPY,
                autoscale=True
                )[0]
        except ConvergenceError as e:
            print(e)
            print("EB failed, setting uniform weights")
            return np.ones(len(P_df))

def run_baselines(P_df, Q_mu, Q_var, baselines = [], seed=0):
    np.random.seed(seed)
    baseline_weights = {}
    if 'raking' in baselines: 
        baseline_weights['raking'] = _run_raking(P_df[Q_mu.columns], Q_mu)[P_df.split=='test'] #makes sure in the same order
    if 'calib' in baselines:
        baseline_weights['calib'] = _run_emp_calib(P_df[Q_mu.columns], Q_mu)[P_df.split=='test'] #makes sure in the same order
    if 'eb' in baselines:
        baseline_weights['eb'] = _run_ent_bal(P_df[Q_mu.columns].copy(), Q_mu, Q_var[Q_mu.columns])[P_df.split=='test'] #makes sure in the same order
    

    
    return baseline_weights
