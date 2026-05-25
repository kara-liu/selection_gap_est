import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from typing import Callable
from dataclasses import dataclass, field, replace, asdict


@dataclass(frozen=True)
class MethodConfig:
    # heuristic mm arguments
    heuristic_bootstrap: int = 700 # Number of bootstrap iters
    heuristic_alpha: int = 0.0005 # alpha level for CI
    heuristic_type: str = 'fast' # Fast or exact 
    heuristic_high_dim: int = None # If not None, will run filtering step to narrow down U* candidates

    cont_density_est: str = 'kde' # if continuous variables, this should be the density estimation method 
    clf: str = 'lr'
    standardize_P: bool = True 
    
    def get_model(self, seed) -> Callable:
        # For binary Y right now; model must support seeding, .fit, and .predict_proba
        clfs = {'xgb': lambda seed: XGBClassifier(objective='binary:logistic', n_estimators=30, eval_metric='logloss', random_state=seed),
                'lrcv': lambda seed: LogisticRegressionCV(cv=5, random_state=seed, max_iter=1000),
                'lr': lambda seed: LogisticRegression(random_state=seed)
                # .... can add more; 
                } 

        assert self.clf in clfs.keys()
        return clfs[self.clf](seed)
    
def design_effect(w: np.ndarray) -> float:
    """
    Kish design effect approximation: E[w^2] / (E[w])^2.
    """
    return (w**2).mean() / w.mean()**2


get_binary_bool = lambda X: ((X==0)|(X==1)).all(axis=0)
