# module/recommender/tabm_model.py
import torch
import numpy as np
from module.recommender.tabm_prep import prepare_tabm_inputs, TabMDataset
from torch.utils.data import DataLoader

import tabm

def make_tabm(n_num, cat_cardinalities, device='cpu', k=10):
    """
    Создаёт экземпляр TabM с нужными параметрами (для регрессии).
    """
    backbone_kwargs = {
        "d_in": n_num,
        "n_blocks": 3,
        "d_block": 256,
        "dropout": 0.3,
        "tabm_init": True,
        "scaling_init": "normal",
        "start_scaling_init_chunks": None,
    }

    model = tabm.TabM(
        cat_cardinalities=cat_cardinalities,
        d_out=1,          # регрессия
        k=k,
        backbone_kwargs=backbone_kwargs,
        start_scaling_init="normal"
    ).to(device)

    return model



def load_model(model, model_path: str, device='cpu'):
    """
    Загружает веса модели TabM
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_tabm(model, X_num: np.ndarray, X_cat: np.ndarray, device='cpu', batch_size=64):
    """
    Делаем предсказания TabM на новых данных
    """
    dataset = TabMDataset(X_num, X_cat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb_num, xb_cat in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            out = model(xb_num, xb_cat)
            preds.append(out.cpu().numpy())

    preds = np.vstack(preds).flatten()
    return preds
