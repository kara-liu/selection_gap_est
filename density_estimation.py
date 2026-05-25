import numpy as np
from itertools import product
import pandas as pd 
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class KDEJoint():
    def __init__(self, bandwidths=None, cv=3, kernel="gaussian"):
        self.bandwidths = np.logspace(-1, 1, 8) if bandwidths is None else bandwidths
        self.cv = cv
        self.kernel = kernel

    def fit(self, X: pd.DataFrame):
        self.columns_ = X.columns.tolist()
        self.scaler_ = StandardScaler().fit(X.values)
        Z = self.scaler_.transform(X.values)
        kde = KernelDensity(kernel=self.kernel)
        self.kde_ = GridSearchCV(kde, {"bandwidth": self.bandwidths}, cv=self.cv, n_jobs=-1).fit(Z).best_estimator_
        # log-Jacobian from standardization (rotation-free)
        self.log_jacobian_ = -np.log(self.scaler_.scale_).sum()
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler_.transform(X[self.columns_].values)
        return self.kde_.score_samples(Z) + self.log_jacobian_

    def sample(self, n: int) -> pd.DataFrame:
        Zs = self.kde_.sample(n, random_state=0)
        Xs = self.scaler_.inverse_transform(Zs)
        return pd.DataFrame(Xs, columns=self.columns_)
    
class GMMJoint():
    """
    Gaussian Mixture Model with (optionally) BIC-based component selection.

    Parameters
    ----------
    n_components : int or tuple
        If int: fixed number of components. If tuple (min_k, max_k), choose K by BIC.
    covariance_type : {"full","tied","diag","spherical"}
    reg_covar : float
        Regularization on covariances for stability.
    """
    def __init__(self, n_components=(1, 10), covariance_type="full", reg_covar=1e-6, random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: pd.DataFrame):
        self.columns_ = X.columns.tolist()
        self.scaler_ = StandardScaler().fit(X.values)
        Z = self.scaler_.transform(X.values)

        if isinstance(self.n_components, tuple):
            min_k, max_k = self.n_components
            best_bic, best_gmm = np.inf, None
            for k in range(min_k, max_k + 1):
                gmm = GaussianMixture(
                    n_components=k, covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar, random_state=self.random_state
                ).fit(Z)
                bic = gmm.bic(Z)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            self.gmm_ = best_gmm
            self.k_ = self.gmm_.n_components
        else:
            self.k_ = int(self.n_components)
            self.gmm_ = GaussianMixture(
                n_components=self.k_, covariance_type=self.covariance_type,
                reg_covar=self.reg_covar, random_state=self.random_state
            ).fit(Z)

        # Jacobian from standardization
        self.log_jacobian_ = -np.log(self.scaler_.scale_).sum()
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler_.transform(X[self.columns_].values)
        return self.gmm_.score_samples(Z) + self.log_jacobian_

    def sample(self, n: int) -> pd.DataFrame:
        Zs, _ = self.gmm_.sample(n)
        Xs = self.scaler_.inverse_transform(Zs)
        return pd.DataFrame(Xs, columns=self.columns_)
    

def _binary_fit_eval_joint_prob(df_fit, df_infer):
        assert ((df_fit==0)|(df_fit==1)).all().all()
        
        df_infer = df_infer.astype(int)
        assert ((df_infer==0)|(df_infer==1)).all().all()


        if df_fit.shape[1]==1:
            return (df_fit.value_counts()/len(df_fit))[np.array([0,1])].to_numpy()[df_infer].squeeze()
        
        df_fit = df_fit.astype('category')

        # Get all possible levels for each variable
        categories = [[0,1] for col in df_fit.columns]
        all_combinations = list(product(*categories)) 
        # Create full index of all combinations
        full_index = pd.MultiIndex.from_tuples(all_combinations, names=df_fit.columns)
        # Count observed frequencies
        counts = df_fit.groupby(list(df_fit.columns),observed=True).size()
        counts = counts.reindex(full_index, fill_value=0)
        total_count = counts.sum()
        prob_table = counts / total_count
        prob_df = prob_table.reset_index()
        prob_df.columns = list(df_fit.columns) + ['probability']
        prob_df = prob_df[prob_df.probability != 0]
        prob_df = prob_df.set_index(list(df_fit.columns))['probability']
        
        df_infer = df_infer.merge(prob_df, how='left', left_on=list(df_fit.columns), right_index=True)
        return df_infer['probability'].fillna(0).to_numpy()


def fit_eval_joint_prob(df_fit, df_infer=None, cont_model='kde'):
    """
    Fit a joint distribution on df_fit and evaluate probability on df_infer.
    Handles binary and continuous data
    """
    is_binary = ((df_fit==0)|(df_fit==1)).all().all()
    if df_infer is None:
        df_infer = df_fit

    if is_binary.all():
        return _binary_fit_eval_joint_prob(df_fit, df_infer)
    elif cont_model == 'kde':
        log_prob = KDEJoint().fit(df_fit).score_samples(df_infer)
        return np.exp(log_prob)
    elif cont_model == 'gmm':
        log_prob = GMMJoint().fit(df_fit).score_samples(df_infer)
        return np.exp(log_prob)
    else:
        raise ValueError(f"Unsupported continuous model type: {cont_model}")



def estimate_densities(
    P_df: pd.DataFrame,
    Q_df: pd.DataFrame,
    Xtilde_cols: List,
    Utilde_cols: List,
    Y_col: str,
    agg_bys: List,
    Ustar_col_dict: Dict[str, List],
    simulate: bool = False,
    cont_density_est: str = 'kde'
) -> pd.DataFrame:
    """
    Estimate probability densities needed downstream (e.g., importance weights / bounds).

    This function computes several joint/marginal probabilities from the observed data,
    including (names reflect the code’s conventions):

      - pUtilde:        p(Utilde) estimated on P_df
      - qUtilde:        qUtilde evaluated at P_df rows , fit on Q_df
      - pXYUtilde:      p(X, Y, Utilde) evaluated at P_df rows
      - pXYgUtilde:     p(X, Y | Utilde) = p(X, Y, Utilde) / p(Utilde)

    Then, for each proposed U* set, it computes:
      - pXYgUtildestar: p(X, Y | Utilde, U*) evaluated at P_df rows

    And produces aggregated bound style quantities using `agg_bys` aggregations.

    Parameters
    ----------
    P_df :
        Biased dataset containing Xtilde_cols, Utilde_cols, Y_col, and any U* columns.
    Q_df :
        Target dataset representing samples, must contain columns in Utilde_cols
        Optionally, if simulate=True, we also know Ustar columns, which are used for method evaluation
    Xtilde_cols : covariates
    Utilde_cols : observed U
    Y_col : outcome
    agg_bys :
        List of aggregation functions; either min or max, depending on metrics (for logloss, will be max)
    Ustar_col_dict :
        Mapping from a label (e.g. 'trueU', 'predUcalib') to a list of columns representing U*.
        Each U* column set must be disjoint from Xtilde_cols, Utilde_cols, and Y_col.
    simulate :
        If True, compute additional densities involving Q (e.g. qXYUtilde, qUtildestar) and
        compute a marginalization used for the 'trueU' case, which will be used in the bound decomp function

    Returns
    -------
    pd.DataFrame
        P_df augmented with computed density columns and aggregate bound columns.
    """
     
    Utilde_cols_rmX = list(np.setdiff1d(Utilde_cols, Xtilde_cols))
    pUtilde = fit_eval_joint_prob(P_df[Utilde_cols],cont_model=cont_density_est)
    qUtilde = fit_eval_joint_prob(Q_df[Utilde_cols], P_df[Utilde_cols],cont_model=cont_density_est)
    pYXUtilde = fit_eval_joint_prob(P_df[Xtilde_cols + Utilde_cols_rmX + [Y_col]],cont_model=cont_density_est)
    densities = pd.DataFrame({
        'pUtilde': pUtilde, 
        'qUtilde': qUtilde, 
        'pXYgUtilde': pYXUtilde / pUtilde,
        'pXYUtilde': pYXUtilde / pUtilde,

    })
    if simulate:
        qYXUtilde = fit_eval_joint_prob(Q_df[Xtilde_cols + Utilde_cols_rmX + [Y_col]],
                            P_df[Xtilde_cols + Utilde_cols_rmX + [Y_col]],cont_model=cont_density_est)
        densities['qXYUtilde'] = qYXUtilde
    densities = pd.concat([P_df.reset_index(drop=True), 
                           densities.reset_index(drop=True)],axis=1)

    # Now compute the weights to estimate various expectations, 
    # including the upper bound on RQ

    for Ustar_name, Ustar_cols in Ustar_col_dict.items():
        assert len(np.intersect1d(Xtilde_cols + Utilde_cols + [Y_col], Ustar_cols)) == 0
        YXUtildestar_Pdf = P_df[Xtilde_cols + Utilde_cols_rmX + Ustar_cols + [Y_col]]
        Utildestar_Pdf = P_df[Utilde_cols + Ustar_cols]
        pYXUtildestar  = fit_eval_joint_prob(YXUtildestar_Pdf,cont_model=cont_density_est)
        pUtildestar = fit_eval_joint_prob(Utildestar_Pdf,cont_model=cont_density_est)
        if simulate: 
            Utildestar_Qdf = Q_df[Utilde_cols + Ustar_cols]
            qUtildestar = fit_eval_joint_prob(Utildestar_Qdf, YXUtildestar_Pdf, cont_model=cont_density_est)
        
        densities['pXYgUtildestar'] = pYXUtildestar/pUtildestar
        
        if Ustar_name == 'trueUstar' and simulate:
            densities['pXYgUtildestar*qUstargUtilde'] = (pYXUtildestar/pUtildestar)*(qUtildestar/qUtilde)
            densities[f'marg_XYgUtilde'] = marginalize(densities[Xtilde_cols + Utilde_cols_rmX + Ustar_cols + [Y_col] + 
                                        ['pXYgUtildestar*qUstargUtilde']].rename(columns={'pXYgUtildestar*qUstargUtilde':'prob'}),
                                        keep_vars=Xtilde_cols + Utilde_cols_rmX + [Y_col], sum_vars=Ustar_cols)
                              
        for agg_by in agg_bys:
            densities[f'ub_{Ustar_name}_XYgUtilde_{agg_by}']= densities[Xtilde_cols + Utilde_cols_rmX + [Y_col] + ['pXYgUtildestar']].groupby(by=Xtilde_cols + 
                                                Utilde_cols_rmX + [Y_col])['pXYgUtildestar'].transform(agg_by)
    if simulate:
        densities=densities.drop(columns=['pXYgUtildestar*qUstargUtilde'])           
    return densities.drop(columns=['pXYgUtildestar'])


def marginalize(df, keep_vars, sum_vars):
    """"
    Marginalize over sum_vars columns Y in P(X, Y)
    X and Y should not overlap; the user must determine the 
    variables to keep and thus to return P(X=keep_vars) after 
    marginalizing out sum_vars. 
    """
    assert len(np.intersect1d(keep_vars, sum_vars)) == 0
    assert 'prob' in df.columns 

    df['X_key'] = df[keep_vars].apply(lambda row: tuple(row),axis=1)
    assert (set(df.columns) == set(keep_vars + sum_vars + ['prob', 'X_key']))

    unique = df.drop_duplicates(subset=list(np.union1d(keep_vars, sum_vars))).drop(columns=sum_vars)
    marg_sum = unique[['X_key', 'prob']].groupby('X_key').sum()
    marg_sum=dict(zip(marg_sum.index, marg_sum.prob))
    return df.X_key.apply(lambda x: marg_sum[x]).to_numpy()

