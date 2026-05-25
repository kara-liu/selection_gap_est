# Selection Gap Estimation Utilities

This repository contains the core implementation and experimental code for estimating
our proposed upper bound on generalization gaps under selection bias for the paper "A Practical Upper Bound on Selection Bias Effects in Medical Prediction Models". Although our method is meant to work in real-world settings with limited target data observability, we provide implementation that supports experiments of simulated selection bias. Because MIMIC and All of Us both are non-public data sources, we provide synthetic data experiments and an example real world case using CDC health data. 

The **primary entry point for users** is the `Generalization_Bound_Estimator` class, found in `gap_est_utils.py`, which supports running the proposed method as well as selected baseline approaches. 

---

## Repository Structure
|------ gap_est_utils.py # Core generalization gap estimation logic

|------ baselines.py # Baseline methods

|------ synth_utils.py # Synthetic data generation and running task selection 

|------ density_estimation.py # Handles density estimation of both binary and continuous data

|------ heuristic_mm_ustar.py # Code for our heuristic moment-matching method

|------ plot_utils.py # A few plotting functions to visualize results 

|------ pred_utils.py # Handles fP prediction model learning and evaluation 

|------ utils.py # A few utilities

We additionally provide two example notebooks: 

`synth_experiment.ipynb` - demomonstrates how to run a simulated selection bias experiment, and how to visualize oracle results when $R_Q$ is known (including bound error decomposition plots, etc. )

`real_experiment.ipynb` - demomonstrates how to use our method in practice when $Q$ has limited observability. As an example, we use CDC health data as the biased distribution $P$ and data frome the 2013 NHANES survey for the target, partially observed $Q$. We also show how to visualize the approximate assumption diagnostics. 


---

## Main Usage

The **main component of this repository** is the `Generalization_Bound_Estimator`,
defined in `gap_est_utils.py`. This class orchestrates data handling, model training,
bound estimation, diagnostics, and optional baseline comparisons.

A typical usage pattern is shown below, where `method_config` is an object defined in `utils.py` that specifies method parameters (i.e., the alpha level of the heuristic): 

```python
from gap_est_utils import Generalization_Bound_Estimator

gapest = Generalization_Bound_Estimator(
    simulate_mode=False,
    method_config=method_config
)

results = gapest.run(
    Xtilde_cols=Xtilde_cols, # Variables to use for prediction
    Y_col=Y_col, # Outcome variable
    Utilde_cols=Utilde_cols, # Observed selection variables 
    seed=seed,
    pred_model=method_config.get_model(seed=seed), # Prediction model f_P
    baselines=... # Optional baselines for comparison ['calib', 'eb', 'raking]
    Q_df=QUtilde_df, # Observed target dataset D_Q
    Q_mu=Q_mu, # First moments under Q
    P_df=P_df, # Observed biased dataset D_P
    Q_var=Q_var, # Second moments under Q
    run_diagnostics=True # Run assumption diagnostics
)
```
The output result is a dataframe, where the columns include the estimated upper bound RQhat of our method, as well as the assumption violation diagnostics, and the baselines if the were input to run. 
