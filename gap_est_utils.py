import numpy as np
import pandas as pd
from itertools import combinations, chain
from typing import Callable, Dict, Iterable, List, Tuple, Optional, Sequence
from utils import * 
from pred_utils import eval_model, train_eval
from density_estimation import * 
from heuristic_mm_ustar import run_heuristic_mm_ustar
from utils import design_effect
from baselines import run_baselines
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def all_Utilde_combinations(Xtilde_cols: List, trueU_cols: List):
    """
    Generates all valid combinations of Utilde columns, ensuring U* has no overlap with Xtilde.
    Only available in simulation_mode.

    Parameters
    ----------
    Xtilde_cols : list
        List of observed Xtilde columns.
    trueU_cols : list
        List of true U columns.

    Returns
    -------
    list
        List of valid U~ combinations.
    """
    #Set of all potential Ustar such that Ustar and Xtilde cannot overlap
    Utilde_must_obs = list(np.intersect1d(trueU_cols, Xtilde_cols))
    pot_obs_Us = np.setdiff1d(trueU_cols, Utilde_must_obs)
    cols = list(chain.from_iterable(combinations(pot_obs_Us, r) for r in range(1,len(pot_obs_Us) + 1)))
    final_cols = [Utilde_must_obs] if len(Utilde_must_obs) else []
    final_cols += [Utilde_must_obs + list(x) for x in cols]
    return final_cols
    

class Generalization_Bound_Estimator:
    """
    Bound estimation method for detecting how selection bias affects 
    ML prediction models 
    """
    def __init__(self, simulate_mode=True, method_config: MethodConfig = MethodConfig()):
        self.simulate_mode = simulate_mode   
        self.method_config = method_config     

    def _clean_metric_df(self, df: pd.DataFrame, metrics: list) -> pd.DataFrame:
        """
        Extracts and organizes specific metrics for 'src_tgt' 
        and 'src_src' for the model f_P trained on D_P.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing evaluation metrics for f_P
        metrics : list
            List of metric names (columns) to extract.

        Returns
        -------
        pd.DataFrame
            DataFrame with evaluation metrics RQ applied on target data 
            and and RP applied on source dataset
        """
        output = {}
        for m in metrics:
            output[f'RP_{m}'] = df[df.eval_distr == 'P'][m].iloc[0]
            if 'Q' in df.eval_distr.values:
                output[f'RQ_{m}'] = df[df.eval_distr == 'Q'][m].iloc[0]
                output[f'RQ_vs_RP_{m}'] = output[f'RQ_{m}'] - output[f'RP_{m}']
        return pd.DataFrame(output, index=[0])

    
    def _get_all_metrics(self,
        base_metric_df: pd.DataFrame,
        densities: pd.DataFrame,
        Xtilde_cols: List[str],
        Y_col: str,
        Ustar_col_dict: Dict[str, List[str]],
        model: Callable,
        scaler: Callable,
        metric_aggby: Dict[str, str],
        baseline_weights: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Calculate all performance metrics relative to true RQ and true RP
        """


        def _update_all_metrics(weight: np.ndarray, method_name: str):
            """
            """
            weighted_eval_metrics = eval_model(P_df_test[Xtilde_cols], P_df_test[Y_col], model, scaler, sample_weight=weight)[0]
            base_metric_df[f'{method_name}_de'] = design_effect(weight)

            for metric, agg_by in metric_aggby.items():
                base_metric_df[f'{method_name}_{metric}'] = weighted_eval_metrics[metric]
                base_metric_df[f'{method_name}_{metric}_vs_RP'] = weighted_eval_metrics[metric] - base_metric_df[f'RP_{metric}']
                if self.simulate_mode:
                    base_metric_df[f'{method_name}_{metric}_vs_RQ'] = weighted_eval_metrics[metric] - base_metric_df[f'RQ_{metric}']

        def _add_pairwise_diffs(pairs: Sequence[Tuple[str, str]], metrics: Iterable[str]) -> None:
            """
            For each (left, right) in pairs, substract the two metrics and add columns: '{left}_vs_{right}_{metric}'.
            """
            for metric in metrics:
                for left, right in pairs:
                    base_metric_df[f"{left}_vs_{right}_{metric}"] = (
                        base_metric_df[f"{left}_{metric}"] - base_metric_df[f"{right}_{metric}"]
                    )
        # To remove bias on the logloss, we only evaluate on the test set. 
        # Note, assumes that all densities are in the same order of samples 
        P_df_test = densities[densities.split=='test']

        # Naive IPPW of Utilde
        ippw_Utilde =  P_df_test['qUtilde'] / P_df_test['pUtilde']    
        _update_all_metrics(weight=ippw_Utilde, method_name='ippw_Utilde')


        # Calculate the baseline RQ using the baseline weights 
        for baseline_name, weight in baseline_weights.items():
            _update_all_metrics(weight=weight, method_name=f'baseline_{baseline_name}')

        # Calculate the bound RQhat based on the density weights 
        for Ustar_name, Ustar_cols in Ustar_col_dict.items(): 
            base_metric_df[f'{Ustar_name}_Ustar_cols'] = str(Ustar_cols)
            for aggby in metric_aggby.values():
                ub_weight = ippw_Utilde * P_df_test[f'ub_{Ustar_name}_XYgUtilde_{aggby}'] / P_df_test['pXYgUtilde']
                _update_all_metrics(weight=ub_weight, method_name=f'ub_{Ustar_name}')
                

        if self.simulate_mode: # Q is fully observed so we can calculate other values, ie. useful for the decomp
            _update_all_metrics(weight=P_df_test['qXYUtilde'] / P_df_test['pXYUtilde'], method_name='ippw_XYUtilde')
            _update_all_metrics(weight=ippw_Utilde * P_df_test['marg_XYgUtilde'] / P_df_test['pXYgUtilde'],
                                method_name='ippw_margXYgUtilde')
            _add_pairwise_diffs(pairs=[['ub_trueUstar','ippw_margXYgUtilde'],
                                    ['ippw_margXYgUtilde', 'ippw_XYUtilde']],
                                metrics=metric_aggby.keys())

        return base_metric_df
        
    def _check_valid_input(self, Q_df, Xtilde_cols, Y_col, U_cols, Utilde_cols, Q_mu, Q_var, P_df,
                      standardize_data, baselines):
        """
        Used to check that .run() has valid inputs
        """
        # basics
        assert len(np.setdiff1d(baselines, ['calib', 'eb', 'raking'])) == 0
        assert Q_mu is not None and len(Q_mu) == 1; "Q_mu must be a single-row result where columns are the features and the value is the mean"
        features = set(Q_mu.columns)


        # features in Q_mu must match all features in dataframe
        if self.simulate_mode:
            assert 'S' in Q_df.columns
            assert (set(Q_df.columns) - set('S')) == features; f"Q_df - Q_mu cols: {np.setdiff1d(Q_df.columns, features)}"
            assert U_cols is not None and set(U_cols).issubset(Q_df.columns)
        else:
            assert set(P_df.columns) == features
        
        assert Y_col in features
        assert set(Xtilde_cols).issubset(features)
        if Utilde_cols is not None:
            assert set(Utilde_cols).issubset(Q_df.columns)
            assert set(Utilde_cols).issubset(features)
            if U_cols is not None:
                assert set(Utilde_cols).issubset(U_cols)
                assert np.intersect1d(Xtilde_cols, np.setdiff1d(U_cols, Utilde_cols)).shape[0] == 0, "Xtilde and Ustar cannot overlap"

        if 'eb' in baselines or standardize_data: #or high dim
            assert Q_var is not None and Q_var.shape == Q_mu.shape 
            assert set(Q_var.columns) == features

    def run(
        self,
        Xtilde_cols: List[str],
        Y_col: str,
        seed: int,
        pred_model: Callable,
        baselines: List[str],
        Q_df: pd.DataFrame,
        Q_mu: pd.DataFrame,
        P_df: Optional[pd.DataFrame] = None,
        Q_var:  Optional[pd.DataFrame] = None,
        Utilde_cols: Optional[List[str]] = None ,
        U_cols: Optional[List[str]] = None,
        metric_aggby: Dict[str, str] = {'logloss': 'max'},
        run_diagnostics: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full estimation the bound estimation method

        The behavior differs slightly between simulated and real-world settings:

        - Simulated setting (`self.simulate_mode=True`):
            * Q_df is fully observed
            * P_df is derived as the subset of Q_df with S == 1.
            * True U* is known and can be evaluated directly.
            * True RQ is available for comparison.

        - Real-world setting (`self.simulate_mode=False`):
            * Only P_df is observed, Q_df for Utilde variables, and first and second moments 
            * RQ is unknown and only upper bounds are reported.
            * U* is inferred via heuristics.

        Parameters
        ----------
        Xtilde_cols :
            List of observed covariates used by the prediction model.
        Y_col :
            Name of the outcome variable.
        seed :
            Random seed controlling train/test splits and stochastic procedures.
        pred_model :
            f_P, the prediction model to learn over the biased data (e.g. a sklearn estimator).
        baselines :
            List of baseline reweighting methods to compare against (e.g. ['raking', 'calib', 'eb']).
        Q_df :
            DataFrame representing the target population distribution Q.
            If simulate_mode: must include all relevant columns (Xtilde, Y, U, S). If real-world: only need Utilde. 
        Q_mu :
            DataFrame of population means under Q.
        P_df :
            DataFrame representing the observed biased distribution P.
            Required when `self.simulate_mode=False`. Ignored otherwise.
        Q_var :
            DataFrame of population variances under Q.
        Utilde_cols :
            Optional list of observed selection variables Utilde. In simulation, this may be
            overridden to enumerate multiple candidate sets for all possible U. 
        U_cols :
            List of true unobserved variables (U). Only used when in simulation mode
            to evaluate oracle bounds.
        metric_aggby :
            Mapping from metric name to aggregation function (max/min) used when computing
            bounds (e.g. {'logloss': 'max', 'accuracy': 'min'}; max denotes upper bound, min 
            denotes lower bound).
        run_diagnostics :
            If True, compute and append approximate assumption diagnostic statistics 
            for common support and conditional independence. 
        also note if self.method_config.standardize_P :
            If True, standardize P data using Q population moments for improved
            numerical stability in density estimation and heuristics.
        Returns
        -------
        pd.DataFrame
            A DataFrame where the columns are all the appropriate metrics for that experiment (1 row
            is one experiment, each different Utilde variable if in simulate_mode, or a single row if 
            in real world setting). Metrics (columns) include:
                - True RP from the prediction model f_P (and RQ, if in simulate)
                - Baseline (i.e. eb, calib, raking) RQhat and design effect of those weights 
                - Bound estimates RQhat under true (if in simulate_mode) and heuristic U* identification strategies, 
                  and those weight's design effects 
                - Optional assumption diagnostic statistics.
                - True bound decomposition (if in simulate_mode)

            Each row corresponds to a different choice of observed Utilde variables.
        """
        np.random.seed(seed)
        self._check_valid_input(Q_df, Xtilde_cols, Y_col, U_cols, Utilde_cols, Q_mu, Q_var, P_df,
                      self.method_config.standardize_P, baselines)
        Q_df = Q_df.copy()
        P_df = P_df.copy() if P_df is not None else None
        Q_mu = Q_mu.copy()
        Q_var = Q_var.copy() if Q_var is not None else None

        ### Step 1: Train prediction model fP on the biased data 
        if self.simulate_mode:
            # Will train-test split Q_df and then train only on the subset P_df with S=1
            # Then will evaluate both RP and RQ for all metrics
            metric_df, model_metadata = train_eval(Q_df, Xtilde_cols, Y_col, 
                                        model=pred_model, seed=seed,
                                        data_distr='Q')
            Q_df['split'] = model_metadata['split']
            P_df = Q_df[Q_df.S==1].copy()

        else:
            # Will only train and eval RP on P_df; RQ is unknown 
            metric_df, model_metadata = train_eval(P_df, Xtilde_cols, Y_col, 
                                    model=pred_model, seed=seed,
                                    data_distr='P')
            P_df['split'] = model_metadata['split']
        metric_df = self._clean_metric_df(metric_df, metric_aggby.keys())


        ### (Optional) Step 2: Normalize data for better stability of heuristic method
        if self.method_config.standardize_P: 
            P_std_df =  P_df.copy()
            P_std_df[Q_mu.columns] = (P_std_df[Q_mu.columns] - Q_mu.to_numpy()) / np.sqrt(Q_var.to_numpy())
            Q_var[Q_var.columns] = 1
            Q_mu[Q_mu.columns] = 0
        else:
            P_std_df = P_df
        dfs: List[pd.DataFrame] = []
        
        # In a simulated setting, we know U and can evaluate our bound over all possible subsets of Utilde in U
        if self.simulate_mode and U_cols is not None:
            Utilde_cols = all_Utilde_combinations(Xtilde_cols, U_cols)
        else:
            Utilde_cols = [Utilde_cols]

        # # (Optional) Step 3: Compute baseline weights (i.e. raking) to compare method to
        baseline_weights = run_baselines(P_std_df, Q_mu=Q_mu, 
                                    Q_var=Q_var, baselines=baselines, seed=seed)
            

        # Step 3: For all Utilde observed (usually just one set in real-world settings)
        #        we calculate the proposed bound weights
        for Utilde_cols in Utilde_cols: 
            Ustar_col_dict: Dict[str,List] = {}

            if self.simulate_mode: # True Ustar identification
                Ustar_col_dict['trueUstar'] = list(np.setdiff1d(U_cols, Utilde_cols)) # true U*

            # Ustar identification using our heuristic search
            # Ustar cannot overlap with Xtilde, Utilde, or Y
            all_Ustar_candidate_cols = np.setdiff1d(P_df.columns, ['S','split'] + list(Utilde_cols) + list(Xtilde_cols) + [Y_col]) 
            Ustar_col_dict['predUstar'], heuristic_weights = run_heuristic_mm_ustar(P_df=P_std_df,Utilde_cols=Utilde_cols,
                                                                all_Ustar_candidate_cols=all_Ustar_candidate_cols,Q_mu=Q_mu,
                                                                config=self.method_config)
            # Calculate all intermediary densities needed for the bound weights 
            P_densities = estimate_densities(
                P_df=P_df, Q_df=Q_df, Xtilde_cols=Xtilde_cols, 
                Utilde_cols=Utilde_cols, Ustar_col_dict=Ustar_col_dict, Y_col=Y_col,
                agg_bys=list(set(metric_aggby.values())), simulate=self.simulate_mode,
                cont_density_est=self.method_config.cont_density_est)

            # Weight the expected loss over P by the bound weights to get RQhat
            df = self._get_all_metrics(
                metric_df.copy(), P_densities,
                Xtilde_cols, Y_col, Ustar_col_dict,
                model_metadata['model'], model_metadata['scalar'],
                metric_aggby, baseline_weights=baseline_weights
            )
            df['Utilde_cols'] = str(Utilde_cols)
            df['Utilde_dim'] = len(Utilde_cols)

            if run_diagnostics:
                diagnostics = self.run_assmp_diag(P_df, Q_df, Utilde_cols, Xtilde_cols, Y_col, Ustar_col_dict['predUstar'], heuristic_weights)
                df = pd.concat([df, pd.DataFrame(diagnostics, index=[0])], axis=1)

            dfs.append(df)
            baseline_weights = {}  # only collect baseline metrics once as it does not depend on Utilde

        return pd.concat(dfs, ignore_index=True)
    
    def run_assmp_diag(self, P_df, Q_df, Utilde_cols, Xtilde_cols, Y_col, predUstar_cols, heuristic_weights: np.ndarray):
        """
        Run the three proposed approximate assumption violation diagnostics
        """
        def _get_S_given_Utilde_prop(X_, Y_):
            S_mdl = LogisticRegression()
            X_ = StandardScaler().fit_transform(X_)
            S_mdl.fit(X_, Y_)
            return S_mdl.predict_proba(X_)[:, 1]
    
        def _get_S_given_Utilde_prop(X_, Y_):
            S_mdl = LogisticRegression()
            X_ = StandardScaler().fit_transform(X_)
            S_mdl.fit(X_, Y_)
            return S_mdl.predict_proba(X_)[:, 1]
        P_df = P_df.copy()
        assert len(P_df) ==  len(heuristic_weights)
        diagnostics = {} 

        ### Test 1: KS Propensity of Utilde 
        # Compute observed propensity 1/P(S=1|Utilde)
        # Can either learn a model or reuse density estimates as pUtilde / qUtilde 
        X = pd.concat([P_df[Utilde_cols], Q_df[Utilde_cols]], axis=0).to_numpy()
        Y = np.concatenate([np.ones(len(P_df)), np.zeros(len(Q_df))], axis=0)
        all_Utilde_propensity = _get_S_given_Utilde_prop(X, Y)
        P_prop = all_Utilde_propensity[:len(P_df)]
        Q_prop = all_Utilde_propensity[len(P_df):]
        ks = ks_2samp(P_prop, Q_prop)
        diagnostics.update({"Utilde_propensity_ks_stat": ks.statistic, 
                                  "Utilde_propensity_ks_pval": ks.pvalue})
    
        ### Test 2: Heuristic weight design effect 
        # We already have the computed heuristic weights, so we just need to calculate the design effect 
        diagnostics.update({"heuristic_w_de":design_effect(heuristic_weights)})

        ### Test 3: Propensity invariance, for conditional independence 
        # First, we need to define the propensity sets, ie. stratify P_df based on the propensity scores p(S=1|Utilde),
        # which we can reuse from above
        n_bins = min(5,len(np.unique(P_prop)))
        if n_bins > 2:
            P_df['prop_set'] = pd.qcut(P_prop,q=n_bins,duplicates="drop")
        else:
            P_df['prop_set'] = P_prop
        all_prop_sets = P_df['prop_set'].unique()
        dict_prop_sets = {s: P_df.loc[P_df["prop_set"] == s, :] for s in all_prop_sets}
        
        # Next, calculate the variance in the logloss of a model trained in one prop set and tested in another prop set. 
        Uhat = list(Utilde_cols) + list(predUstar_cols)
        model = LogisticRegression()
        model_ll_perf = []

        for prop_set, df_prop_set in dict_prop_sets.items():
            Uhat_data = df_prop_set[Uhat].to_numpy()
            
            for V in Xtilde_cols + [Y_col]: 
                # learn p(V|Uhat,S*=s) for all s, for all V
                Xtr, Xtst, ytr, ytst = train_test_split(Uhat_data, df_prop_set[V], test_size=.2, random_state=42)
                if len(np.unique(ytr)) < 2 or len(np.unique(ytst)) < 2:
                    continue
                model.fit(Xtr, ytr)
                # score model on its own test split + on all other propensity strata
                scores = [model.score(Xtst, ytst)] + [
                    model.score(dict_prop_sets[prop_set_2][Uhat], dict_prop_sets[prop_set_2][V])
                    for prop_set_2 in dict_prop_sets.keys()
                    if prop_set_2 != prop_set
                ]

                model_ll_perf.append(np.var(scores))
        diagnostics.update({"propinv_ci_avg_var": np.mean(model_ll_perf)})
        return diagnostics
  
    


