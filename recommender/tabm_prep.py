# module/recommender/tabm_prep.py
import torch
from torch.utils.data import Dataset


def prepare_tabm_inputs(df):
    """
    Подготовка данных для TabM.

    Возвращает:
        X_num: torch.Tensor с числовыми признаками
        X_cat: torch.Tensor с категориальными признаками (индексы)
        y: torch.Tensor с метками
        feature_names: список названий признаков
        cat_cardinalities: список количества уникальных значений категориальных признаков
    """
    df = df.copy()

    # Определяем целевую колонку
    target_col = "performance_score"

    # Числовые признаки
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Категориальные признаки
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Кодируем категориальные в индексы
    cat_cardinalities = []
    X_cat = []
    for col in cat_cols:
        df[col] = df[col].astype("category")
        cat_cardinalities.append(len(df[col].cat.categories))
        X_cat.append(df[col].cat.codes.values)

    X_cat = torch.tensor(X_cat, dtype=torch.long).T if X_cat else torch.empty((len(df), 0), dtype=torch.long)
    X_num = torch.tensor(df[num_cols].values, dtype=torch.float32) if num_cols else torch.empty((len(df), 0),
                                                                                                dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    return X_num, X_cat, y, num_cols + cat_cols, cat_cardinalities


class TabMDataset(Dataset):
    """
    Dataset для TabM. Возвращает батчи числовых и категориальных признаков и метки.
    """

    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
