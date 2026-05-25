from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    r2_score,
    mean_squared_error
)

# lr_cv = lambda: LogisticRegressionCV(cv=5, random_state=0, n_jobs=-1, max_iter=1000, class_weight='balanced')
lr_cv_ = lambda s: LogisticRegressionCV(cv=5, random_state=s, n_jobs=-1, max_iter=1000, class_weight='balanced')
# lr_cv_nbal = lambda: LogisticRegressionCV(cv=5, random_state=0, n_jobs=-1, max_iter=1000)
# lr_cv_nbal_ = lambda s: LogisticRegressionCV(cv=5, random_state=s, n_jobs=-1, max_iter=1000)
# lr_cv_nbal_nw_= lambda s: LogisticRegressionCV(cv=5, random_state=s,  max_iter=1000)
# lr_nopen_bal = lambda: LogisticRegression(n_jobs=-1, class_weight='balanced')
# lasso = lambda: LassoCV(cv=5, random_state=0, n_jobs=-1)

def eval_metric(name):
    if name == 'logloss':
        return lambda y_true, y_pred: log_loss(y_true, y_pred)
    elif name == 'auc':
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    elif name == 'acc':
        return lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(int))
    raise ValueError(f'Unknown metric {name}')


def _is_classifier(model) -> bool:
    """Heuristic: treat as classifier if predict_proba exists."""
    return hasattr(model, "predict_proba")


# Similar metrics for regression could be defined 
def _eval_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    return {
        "acc": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "auc": roc_auc_score(y_true, y_prob, sample_weight=sample_weight),
        "prc": average_precision_score(y_true, y_prob, sample_weight=sample_weight),
        "brier": brier_score_loss(y_true, y_prob, sample_weight=sample_weight),
        "logloss": log_loss(y_true, y_prob, sample_weight=sample_weight),
    }



def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    sample_weight: Optional[np.ndarray] = None,
    standardize: bool = True,
):
    model = clone(model)
    scaler = StandardScaler().fit(X_train) if standardize else None
    Xtr = scaler.transform(X_train) if scaler is not None else X_train
    model.fit(Xtr, y_train, sample_weight=sample_weight)
    return model, scaler


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }

def eval_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model,
    scaler,
    sample_weight: Optional[np.ndarray] = None,
):
    Xt = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(Xt)

    if _is_classifier(model):
        y_prob = model.predict_proba(Xt)[:, 1]
        metrics = _eval_classification(y_test, y_pred, y_prob, sample_weight=sample_weight)
        return metrics, y_prob
    else:
        metrics = _eval_regression(y_test, y_pred)
        return metrics, None

def train_eval(
    df: pd.DataFrame,
    X_cols: Sequence[str],
    y_col: str,
    model: Callable,
    data_distr: str ='P',
    seed: int = 0,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train on the biased/source distribution P and evaluate on P (and optionally Q).

    This is designed to support both when you only have access to the biased dataset P, 
    and also when you have access to the full target dataset Q with a selection indicator S.

    Parameters
    ----------
    df:
        Input dataframe containing feature columns and label column.
        If `data_distr="Q"`, it must also contain `S` indicating membership in P.
    X_cols:
        Feature column names used to train/evaluate the model.
    y_col:
        Label column name.
    model:
        Any scikit-learn style estimator.
    data_distr:
        - "P": treat `df` as the biased/source dataset only.
        - "Q": treat `df` as the full target dataset, with `df[S]==1` indicating P rows.
    seed:
        Random seed controlling the train/test split.
    standardize:
        If True, fit a StandardScaler on the training features and apply to eval features.

    Returns
    -------
    metrics_df:
        A dataframe of metric performance of the model eval'd on P (and optionally Q). 
    metadata:
        Dictionary containing:
          - "split": np.ndarray of shape (len(df),) with values {"train","test"}
          - "model": fitted estimator
          - "scalar": fitted StandardScaler or None
    """
    assert data_distr in ['P', 'Q']
    eval_metrics = {}

    n = len(df)
    idx_all = np.arange(n)
    idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=seed)

    split = np.empty(n, dtype=object)
    split[:] = ""
    split[idx_train] = "train"
    split[idx_test] = "test"

    metadata: Dict[str, Any] = {
        "split": split,
    }

    if data_distr == 'Q':
        assert 'S' in df.columns
        P_tr = df[(split == 'train') & (df.S==1)]
        P_tst = df[(split == 'test') & (df.S==1)]
        Q_tst = df[split == 'test'] 
    else:
        P_tr = df[split == 'train']
        P_tst = df[split == 'test']

    # Fit on biased dataset P
    X_P_tr = P_tr[X_cols].to_numpy()
    y_P_tr = P_tr[y_col].to_numpy()
    model_fit, scalar = fit_model(X_P_tr, y_P_tr, model, standardize=standardize)
    metadata.update({"model": model_fit, "scalar": scalar})

    # Eval on biased test set P
    X_P_tst = P_tst[X_cols].to_numpy()
    y_P_tst = P_tst[y_col].to_numpy()
    eval_metrics['P'] = eval_model(X_P_tst, y_P_tst,  model_fit, scalar)[0] #TODO: fix if don't use yprob
    
    if data_distr == 'Q':
        # Eval on biased test set Q
        X_Q_tst = Q_tst[X_cols].to_numpy()
        y_Q_tst = Q_tst[y_col].to_numpy()
        eval_metrics['Q'] = eval_model(X_Q_tst, y_Q_tst,  model_fit, scalar)[0]

    return pd.DataFrame.from_dict(eval_metrics, orient='index').reset_index(names='eval_distr'), metadata
    


def get_xgb_cv(which='clf', n_est=[25,50], params={}, verbose=1):
    assert which in ['clf','rgr']
    if which=='clf':
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    else:
        model = XGBRegressor()

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': n_est,  
        'max_depth': [3, 5],  
        'learning_rate': [0.01, 0.1], 
    }
    param_grid.update(params)
    

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  
        verbose=verbose,
        n_jobs=-1
    )
    return grid_search


