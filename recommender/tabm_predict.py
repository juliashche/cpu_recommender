"""
Loads saved TabM model and preprocessors and predicts target for given DataFrame.
Usage:
    from module.recommender.tabm_predict import predict_with_tabm
    preds = predict_with_tabm(df)  # returns numpy array same length as df
"""

import joblib
import numpy as np
import pandas as pd
import torch

# import TabM factory like in trainer
try:
    from tabm import TabM
    def make_tabm(n_num, cat_cardinalities, device='cpu'):
        return TabM.make(n_num_features=n_num, cat_cardinalities=cat_cardinalities, d_out=1)
except Exception:
    try:
        from tabm.model import TabM
        def make_tabm(n_num, cat_cardinalities, device='cpu'):
            return TabM(n_num_features=n_num, cat_cardinalities=cat_cardinalities, d_out=1)
    except Exception as e:
        raise ImportError("Cannot import TabM. Install 'tabm' package.") from e

import os
MODEL_FILE = "models/tabm_small.pth"
SCALER_FILE = "models/tabm_scaler.joblib"
CAT_MAPS_FILE = "models/tabm_cat_maps.joblib"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _apply_cat_maps(df, cat_maps):
    # map categorical columns by saved maps; unseen -> new index appended
    df = df.copy()
    cat_cols = list(cat_maps.keys())
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        series = df[c].fillna("<NA>").astype(str)
        mapping = cat_maps[c]
        # unseen -> add index = len(mapping) (will be out of cardinalities expected by model, but TabM can accept)
        X_cat[:, i] = series.map(lambda v: mapping.get(v, len(mapping))).astype(np.int64)
    return X_cat, cat_cols

def predict_with_tabm(df: pd.DataFrame) -> np.ndarray:
    # load preproc
    scaler = joblib.load(SCALER_FILE)
    cat_maps = joblib.load(CAT_MAPS_FILE)

    # numeric columns used by scaler: we don't store names in file; assume trainer used all numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # but scaler expects same order/number as during training; in our trainer we used all numeric columns from df_train at training
    # so we must ensure columns align. Best practice: train and predict on same raw dataframe layout.
    X_num = scaler.transform(df[num_cols].fillna(0.0).astype(float))

    X_cat, cat_cols = _apply_cat_maps(df, cat_maps)

    # build model skeleton
    n_num = X_num.shape[1]
    cat_cardinalities = [len(cat_maps[c]) for c in cat_cols]

    model = make_tabm(n_num=n_num, cat_cardinalities=cat_cardinalities, device=DEVICE)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    # dataloader
    ds = torch.utils.data.TensorDataset(torch.tensor(X_num, dtype=torch.float32),
                                        torch.tensor(X_cat, dtype=torch.long) if X_cat.size else torch.zeros((len(X_num),0),dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=256)

    preds = []
    with torch.no_grad():
        for xb_num, xb_cat in loader:
            xb_num = xb_num.to(DEVICE)
            xb_cat = xb_cat.to(DEVICE)
            p = model(x_num=xb_num, x_cat=xb_cat).cpu().numpy().ravel()
            preds.append(p)
    if preds:
        preds = np.concatenate(preds)
    else:
        preds = np.array([])

    return preds
