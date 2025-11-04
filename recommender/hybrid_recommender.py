import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
from module.recommender.tabm_model import load_model, predict_tabm
from module.recommender.tabm_prep import prepare_tabm_inputs
from module.recommender.tabm_model import make_tabm


def content_based(df: pd.DataFrame, target_model: str, top_k: int = 5) -> pd.DataFrame:
    """Контент-базированные рекомендации с расширенным набором признаков."""
    feature_cols = [
        'clock_speed',
        'tdp',
        'performance_score',
        'perf_per_watt',
        'perf_per_ghz',
        'mem_efficiency'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    features = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    sim = cosine_similarity(scaled)

    matches = df[df['cpu_model'].str.contains(target_model, regex=False)]
    if matches.empty:
        raise ValueError(f"Target model '{target_model}' не найдена в датафрейме.")

    idx = matches.index[0]
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k + 1]
    similar_indices = [i for i, _ in sim_scores]

    result = df.iloc[similar_indices][['cpu_model'] + feature_cols].copy()
    result['content_score'] = np.linspace(1.0, 0.5, len(result))
    return result


def tabm_based(df: pd.DataFrame, model_path: str, top_k: int = 5) -> pd.DataFrame:
    """Использует обученную TabM для вычисления скоринга."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_num, X_cat, y, _, cat_cardinalities = prepare_tabm_inputs(df)
    model = make_tabm(n_num=X_num.shape[1], cat_cardinalities=cat_cardinalities, device=device)
    model = load_model(model, model_path, device=device)

    preds = predict_tabm(model, X_num, X_cat, device=device)
    df_tabm = df[['cpu_model']].copy()
    # Нормируем предсказания для объединения с content_score
    df_tabm['tabm_score'] = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)
    return df_tabm.sort_values('tabm_score', ascending=False).head(top_k)


def hybrid(df: pd.DataFrame, target_model: str, model_path: str,
           alpha: float = 0.5, beta: float = 0.5, top_k: int = 5) -> pd.DataFrame:
    """
    Гибридный метод: объединяет только content-based и TabM-предсказания.
    """
    content = content_based(df, target_model, top_k)
    tabm = tabm_based(df, model_path, top_k)

    merged = content.merge(tabm, on='cpu_model', how='outer').fillna(0)

    # Итоговый скор — усреднение двух компонентов
    merged['hybrid_score'] = alpha * merged['content_score'] + beta * merged['tabm_score']

    return merged.sort_values('hybrid_score', ascending=False).head(top_k)
