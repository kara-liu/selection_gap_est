import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from gap_est_utils import Generalization_Bound_Estimator, MethodConfig
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from pred_utils import eval_metric
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Callable, Optional, Any
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import brentq
from numpy.linalg import eigvals
from scipy import stats
from scipy import special
import copy
from dataclasses import dataclass, field, replace, asdict

@dataclass(frozen=True)
class SynthDataConfig:
    # top-level
    n: int = 10_000 # Sample size of DQ
    n_tasks: int = 1 # Prediction task + selection mechanism; each task iterate over all Utilde \subset U possible
    n_seeds: int = 1 # Number of seeds per task; each seed will resample S and relearn fP

    min_gap: float = 0.03 # Mimimum generalization gap to proceed with a task, RQ - RP > min_gap

    # schemas
    U_schema: Dict[str, Any] = field(default_factory=lambda: {
        "normal": 0, "uniform": 0, "bernoulli": 3, "corr_U": 0.15
    }) 
    # X = (Xtilde, Xother) 
    X_schema: Dict[str, int] = field(default_factory=lambda: {"binary": 7, "cont": 0})
    C_schema: Dict[str, int] = field(default_factory=lambda: {"binary": 6, "cont": 0})
    C_type: str = 'independent' # 'colliders' or 'independent' 

    dXtilde: int = 5  # should be less than dX
    
    # degrees / dims in polynomial functions generating X, Y, S
    deg_X: int = 2
    deg_Y: int = 2
    deg_C: int = 2
    deg_S: int = 2 # 1 = linear; 2>= nonlinear selection mechanism

    # Std of these variables
    beta_std_S: float = 2.0 # selection strength parameter
    beta_std_X: float = 2.0
    beta_std_Y: float = 2.0
    beta_std_C: float = 1.0

    # model selection
    clf: str = "xgb"
    metric: str = "logloss" # l(), which determines the expected risk E[l(Y, fp(Xtilde))]

    @property
    def dX(self) -> int:
        dX = sum(self.X_schema.values())
        assert dX >= self.dXtilde
        return dX
    
    def to_kwargs(self) -> dict:
        return asdict(self)
    
    def get_model(self, seed) -> Callable:
        # For binary Y right now; model must support seeding, .fit, and .predict_proba
        clfs = {'xgb': lambda seed: XGBClassifier(objective='binary:logistic', n_estimators=30, eval_metric='logloss', random_state=seed),
                'lrcv': lambda seed: LogisticRegressionCV(cv=5, random_state=seed, max_iter=1000),
                'lr': lambda seed: LogisticRegression(random_state=seed)
                # .... can add more; 
                } 

        assert self.clf in clfs.keys()
        return clfs[self.clf](seed)

    def get_eval_metric(self, a, b) -> Callable:
        return eval_metric(self.metric)(a,b)
    
class SyntheticDataGenerator:
    """
    Wrapper that holds a synth_config and uses it across simulate_* calls.

    Will sample and save the params (ie. the betas) for reuse such that we 
     can sample from the same data distribuiton again with a different seed 
    """

    def __init__(self, synth_config: SynthDataConfig):
        self.synth_config = synth_config
        self.beta_C = None
        self.beta_Y = None
        self.beta_X = None
        self.beta_S = None
        self.Xtilde_idxs = None

    def simulate_C(self,
        X: np.ndarray,
        Y: np.ndarray,
        rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate extra variables C. Either colliders or 
        independent variables. 

        """    
        if rng is None:
            rng = np.random.default_rng()
        nC = sum(self.synth_config.C_schema.values())

        qc = int(self.synth_config.C_schema.get('cont', 0))
        qb  = int(self.synth_config.C_schema.get('binary', 0))
        assert (qc == nC) ^ (qb == nC)

        if self.synth_config.C_type == 'colliders':
            XY = np.hstack([X, Y.reshape(-1, 1)])
            if self.synth_config.deg_C is not None:
                poly = PolynomialFeatures(degree=self.synth_config.deg_C, include_bias=False)
                XY = poly.fit_transform(XY)

            if self.beta_C is None:
                beta = rng.normal(0, 1, size=(XY.shape[1], nC))
                beta *=  (self.synth_config.beta_std_C / (XY @ beta).std() + 1e-12)
                self.beta_C = beta
            elif self.beta_C.shape != (XY.shape[1], nC):
                raise ValueError(f"beta_cont must have shape {(XY.shape[1], nC)}")
            eps_c = rng.normal(0, 0.1, size=(XY.shape[0], nC))
            C = XY @ self.beta_C + eps_c
        
        else:
            C = rng.normal(0, 1, size=(X.shape[0], nC))

        if qb: 
            C = rng.binomial(1, expit(C))

        return C

    def simulate_U(
        self,
        rng: np.random.Generator = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        schema = copy.copy(self.synth_config.U_schema)
        corr_U = schema.pop('corr_U')
        setdiff=np.setdiff1d(list(schema.keys()), ['uniform', 'bernoulli', 'normal'])
        assert len(setdiff) == 0, f"Unexpected U vars {setdiff}"
            
        qn = schema.get('normal', 0)
        qu = schema.get('uniform', 0)
        qb = schema.get('bernoulli', 0)
        if corr_U > 0:
            assert qn == sum(schema.values()) or qb == sum(schema.values()), "Cannot have a mixed type or uniform correlated U right now"
            
        parts = []
        if corr_U == 0:
            normal_params = [(0, 1)] * qn
            for i, (mu, sd) in enumerate(normal_params):
                if sd <= 0:
                    raise ValueError(f"Standard deviation must be positive for normal variable {i}, got {sd}")
                x = rng.normal(mu, sd, size=(self.synth_config.n, 1))
                parts.append(x)
        elif qn > 0:
            exit = True
            while exit: 
                try:
                    parts = [simulate_correlated_binary(self.synth_config.n, 
                            np.random.uniform(.2,.8, size=qn), 
                            get_R(qn, corr_U), binarize=False, rng=rng)]  
                    exit = False
                except ValueError:
                    pass
            
        uniform_params = [(0, 1)] * qu
        for i, (low, high) in enumerate(uniform_params):
            uniform_params = uniform_params or [(0, 1)] * qu
            if high <= low:
                raise ValueError(f"High must be greater than low for uniform variable {i}, got low={low}, high={high}")
            x = rng.uniform(low, high, size=(self.synth_config.n, 1))
            parts.append(x)
            
        if corr_U == 0:
            bernoulli_params = [0.5] * qb
            for i, p in enumerate(bernoulli_params):
                if not (0 <= p <= 1):
                    raise ValueError(f"Probability must be in [0, 1] for bernoulli variable {i}, got {p}")
                x = rng.binomial(1, p, size=(self.synth_config.n, 1)).astype(int)
                parts.append(x)
        elif qb > 0:
            exit = True

            while exit: 
                try:
                    parts = [simulate_correlated_binary(self.synth_config.n, 
                        np.random.uniform(.2,.8, size=qb), get_R(qb, corr_U), \
                            binarize=True, rng=rng).astype(int)]  
                    exit = False
                except ValueError:
                    pass
                    
        if not parts:
            raise ValueError("No variables to simulate. Please provide a valid schema or parameters.")

        U = np.hstack(parts)
        return U

    def simulate_X(self,
        U: np.ndarray,
        rng: np.random.Generator = None
    ):
        schema = copy.copy(self.synth_config.X_schema)
        setdiff=np.setdiff1d(list(schema.keys()), ['cont', 'binary'])
        dX = self.synth_config.dX
        assert len(setdiff) == 0, f"Unexpected X vars {setdiff}"
        if rng is None:
            rng = np.random.default_rng()

        poly = PolynomialFeatures(degree=self.synth_config.deg_X, include_bias=False)
        U_poly = poly.fit_transform(U)

        if self.beta_X is None:
            beta = rng.normal(0, 1, size=(U_poly.shape[1], dX))
            beta *= (self.synth_config.beta_std_X / ((U_poly @ beta).std() + 1e-12))
            self.beta_X = beta
        else:
            if self.beta_X.shape != (U_poly.shape[1], dX):
                raise ValueError(f"Beta shape must be {(U_poly.shape[1], dX)}, got {beta.shape}")
        
        eps = rng.normal(0, 0.1, size=(U.shape[0], dX))
        X = U_poly @ self.beta_X + eps
        if schema.get('binary', 0) > 0:
            pX = special.expit(X[:,:schema['binary']]).squeeze()
            X[:,:schema['binary']] = rng.binomial(1, pX)
        return X

    def simulate_Y(
        self,
        U: np.ndarray,
        X: np.ndarray,
        rng: np.random.Generator = None
    ):
        
        if rng is None:
            rng = np.random.default_rng()

        UX = np.hstack([U, X])
        if self.synth_config.deg_Y is not None:
            poly = PolynomialFeatures(degree=self.synth_config.deg_Y, include_bias=False)
            UX = poly.fit_transform(UX)
            
        if self.beta_Y is None:
            beta = rng.normal(0,  1, size=(UX.shape[1],))
            beta *= (self.synth_config.beta_std_Y / ((UX @ beta).std() + 1e-12))
            self.beta_Y = beta
        else:
            if self.beta_Y.shape != (UX.shape[1],):
                raise ValueError(f"Beta shape must be {(UX.shape[1],)}, got {beta.shape}")

        eps = rng.normal(0, .1, size=(U.shape[0],))
        logits = UX @ self.beta_Y + eps
        Y = rng.binomial(1, expit(logits), size=U.shape[0])
        return Y

    def simulate_S(
        self,
        U: np.ndarray,
        rng: np.random.Generator = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        poly = PolynomialFeatures(degree=self.synth_config.deg_S, include_bias=False)
        U_poly = poly.fit_transform(U)

        if self.beta_S is None:
            beta = rng.normal(0, 1, size=(U_poly.shape[1],))
            beta *= (self.synth_config.beta_std_S / ((U_poly @ beta).std() + 1e-12))
            self.beta_S = beta
        else:
            if self.beta_S.shape != (U_poly.shape[1],):
                raise ValueError(f"Beta shape must be {(U_poly.shape[1],)}, got {beta.shape}")
        eps = 0 
        logits = U_poly @ self.beta_S  + eps
        pS = expit(logits)
        S = rng.binomial(1, pS, size=U.shape[0])
        return S

    def simulate_data(
        self,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        if self.Xtilde_idxs is None:
            self.Xtilde_idxs = rng.choice(np.arange(self.synth_config.dX), 
                                           size=self.synth_config.dXtilde, replace=False)
        U = self.simulate_U(rng=rng)
        X = self.simulate_X(U, rng=rng)
        Y = self.simulate_Y(U, X[:,self.Xtilde_idxs],rng=rng)
        S = self.simulate_S(U, rng=rng)
        C = None

        if self.synth_config.C_schema is not None:
            C = self.simulate_C(X=X[:, self.Xtilde_idxs],Y=Y, rng=rng
                )
        return U, X, Y, S, C

    def sample_params_get_gap(
        self,
        seed: int
    ):
        """
        Given the config, sample parameters and compute the RQ - RP gap.
        """
        _, X, Y, S, _ = self.simulate_data(seed=seed)
        Xtilde = X[:, self.Xtilde_idxs]
        mdl = self.synth_config.get_model(seed=seed)
        mdl.fit(Xtilde[S==1], Y[S==1])
        Y_pred_prob = mdl.predict_proba(Xtilde)[:, 1]
        gap = self.synth_config.get_eval_metric(Y, Y_pred_prob) - self.synth_config.get_eval_metric(Y[S==1], Y_pred_prob[S==1])
        return gap


def get_R(dU, off_diag):
    R = np.eye(dU)
    R[np.triu_indices(dU, k=1)] = off_diag
    R[np.tril_indices(dU, k=-1)] = off_diag
    return R

def latent_corr_from_binary(p_i, p_j, rho_target, tol=1e-6):
    """Solve for latent normal correlation that yields rho_target binary correlation."""
    # Define thresholds
    a_i, a_j = stats.norm.ppf(p_i), stats.norm.ppf(p_j)

    def f(rho_lat):
        p_ij = stats.multivariate_normal(cov=[[1, rho_lat], [rho_lat, 1]]).cdf([a_i, a_j])
        rho = (p_ij - p_i * p_j) / np.sqrt(p_i*(1-p_i)*p_j*(1-p_j))
        return rho - rho_target

    return brentq(f, -0.99999, 0.99999, xtol=tol)

def get_attainable_corr(p_i, p_j):
    min_corr = max(-np.sqrt(p_i * p_j / (1 - p_i) / (1 - p_j)), -np.sqrt((1 - p_i) * (1 - p_j) / p_i / p_j))
    max_corr = min(np.sqrt(p_i * (1 - p_j) / (1 - p_i) / p_j), np.sqrt((1 - p_i) * p_j / p_i / (1 - p_j)))
    return min_corr, max_corr

def nearest_pd(A, eps=1e-8):
    """Nearest positive-definite matrix."""
    B = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < eps] = eps
    return (eigvec * eigval) @ eigvec.T

def simulate_correlated_binary(n, p, R, rng, binarize=True):
    """
    Draw n samples of k correlated Bernoulli variables.

    Parameters
    ----------
    n : int
        Number of rows (observations).
    p : array-like, shape (k,)
        Desired marginal probabilities.
    R : array-like, shape (k,k)
        Target Pearson correlation matrix for the binary variables (ones on diag).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    U : ndarray, shape (n,k)
        Simulated 0/1 sample.
    """
    p = np.asarray(p)
    R = np.asarray(R)
    k = len(p)
    assert R.shape == (k, k), "R must be k×k"

    # Clip correlation matrix to attainable range
    for i in range(k):
        for j in range(i + 1, k):
            min_corr, max_corr = get_attainable_corr(p[i], p[j])
            R[i, j] = np.clip(R[i, j], min_corr, max_corr)

    # Build latent correlation matrix
    Sigma = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rho_star = latent_corr_from_binary(p[i], p[j], R[i, j])
            Sigma[i, j] = Sigma[j, i] = rho_star

    # Ensure PSD
    if np.min(eigvals(Sigma)) < 1e-8:
        Sigma = nearest_pd(Sigma)

    # Draw latent normals
    Z = rng.multivariate_normal(mean=np.zeros(k), cov=Sigma, size=n)

    if binarize:
        # Threshold to binary
        thresholds = stats.norm.ppf(p)
        U = (Z < thresholds).astype(int)
    else:
        U = Z
    return U



def run_synth_random(synth_config: SynthDataConfig, 
                     method_config: MethodConfig,
                     baselines=['calib', 'raking', 'eb'],
                    seed_start=-1):
    """
    Run synthetic experiments over multiple tasks, searching for parameters that
    induce a large enough RQ-RP gap, then simulate data and evaluate 
    generalization bound estimation methods (plus baselines)

    Parameters
    ----------
    synth_config : SynthDataConfig
        Experiment configuration.

    baselines : sequence of str, optional
        Which baselines evaluate. Default
        ('calib', 'raking', 'eb').

    Returns
    -------
    pd.DataFrame
        Concatenated results across successful tasks. Empty DataFrame if none.
    """


    results = []
    total_tasks = 0
    i = seed_start

    pbar = tqdm(total=synth_config.n_tasks, desc="Processing tasks")

    while total_tasks < synth_config.n_tasks:
        i+=1


        # Sample parameters corresponding to one task 
        # Then sample all other params and compute gap 
        data_gen = SyntheticDataGenerator(synth_config)
        gap = data_gen.sample_params_get_gap(seed=i)
        #If gap RQ - RP too low (good generalization already), skip 
        if gap < synth_config.min_gap:
            continue

        # Repeat across seeds
        for seed in range(i, i + synth_config.n_seeds):

            U, X, Y, S, C = data_gen.simulate_data(seed=seed)

            # Assemble dataframe
            X_cols = [f'X{k}' for k in range(X.shape[1])]
            U_cols = [f'U{k}' for k in range(U.shape[1])]
            Xtilde_cols = list(np.array(X_cols)[data_gen.Xtilde_idxs])
            Q_df = pd.concat(
                [
                    pd.DataFrame(X, columns=X_cols),
                    pd.DataFrame(Y, columns=['Y']),
                    pd.DataFrame(U, columns=U_cols),
                    pd.DataFrame(S, columns=['S'])
                ],
                axis=1,
            )
            if C is not None: # C = variables that are uncorrelated with other variables, OR are colliders
                C_df = pd.DataFrame(C, columns=[f'C{k}' for k in range(C.shape[1])])
                Q_df = pd.concat([Q_df, C_df], axis=1)
            Q_mu = Q_df.drop(columns='S').mean(axis=0).to_frame().T
            Q_var = Q_df.drop(columns='S').var(axis=0).to_frame().T
            
            gapest = Generalization_Bound_Estimator(simulate_mode=True,method_config=method_config)
            df = gapest.run(Xtilde_cols=Xtilde_cols, Y_col='Y', U_cols=U_cols, seed=seed, 
                            pred_model=synth_config.get_model(seed=seed), 
                            baselines=baselines, Q_df=Q_df, Q_mu=Q_mu, 
                            Q_var=Q_var)
            
            results.append(df.assign(task=i,seed=seed))
        
        total_tasks += 1
        pbar.update(1)
    results = pd.concat(results)
    return results
