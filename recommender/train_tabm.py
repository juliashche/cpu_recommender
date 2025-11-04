import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import tabm

train_path = "data/raw/SPR/train_data.csv"
df_train = pd.read_csv(train_path)

# Генерация синтетических данных (если нужно)
def generate_synthetic(df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = len(df)
    synth = df.sample(n_samples, replace=True).reset_index(drop=True)
    for col in df.select_dtypes(include=np.number).columns:
        synth[col] = synth[col].astype(float) * np.random.uniform(0.85, 1.15, len(synth))
    for col in df.select_dtypes(include=object).columns:
        synth[col] = np.random.choice(df[col].unique(), len(synth))
    return synth

df_synth = generate_synthetic(df_train, n_samples=len(df_train)*3)
df_train_full = pd.concat([df_train, df_synth], ignore_index=True)

target_col = "RESULT.Value"
feature_cols = [c for c in df_train_full.columns if c != target_col]

X = df_train_full[feature_cols].copy()
y = df_train_full[target_col].values.astype(np.float32)

# Категориальные → индексы
for col in X.select_dtypes(include=object).columns:
    X[col] = X[col].astype("category").cat.codes

X = torch.from_numpy(X.values.astype(np.float32))
y = torch.from_numpy(y).unsqueeze(-1)

d_in = X.shape[1]
k = 10
model = nn.Sequential(
    tabm.EnsembleView(k=k),
    tabm.MLPBackboneBatchEnsemble(d_in=d_in, n_blocks=3, d_block=256, dropout=0.3,
                                  k=k, tabm_init=True, scaling_init='normal', start_scaling_init_chunks=None),
    tabm.LinearEnsemble(256, 1, k=k)
)

criterion = nn.HuberLoss(delta=0.5, reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

train_dataset = TensorDataset(X, y)  # создаём TensorDataset отдельно
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

n_epochs = 20
model.train()

for epoch in range(n_epochs):
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)  # shape: (batch, k, 1)

        # Усреднение по ансамблю
        preds_mean = preds.mean(dim=1)

        # Вычисление лосса
        loss = criterion(preds_mean, yb)
        loss.mean().backward()
        optimizer.step()

        total_loss += loss.sum().item()

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "tabm_ensemble_model.pth")
print("Модель обучена и сохранена.")


test_path = "data/raw/SPR/test_with_preds.csv"
df_test = pd.read_csv(test_path)

# Категориальные признаки → индексы (совпадающие с train)
for col in df_test.select_dtypes(include=object).columns:
    if col in df_train_full.columns:
        cat_map = {cat: code for code, cat in enumerate(df_train_full[col].astype("category").cat.categories)}
        df_test[col] = df_test[col].map(cat_map).fillna(0).astype(int)

# Преобразуем в тензор
X_test = torch.from_numpy(df_test[feature_cols].values.astype(np.float32))


model.eval()
with torch.no_grad():
    preds_test = model(X_test)
    preds_mean = preds_test.mean(dim=1).numpy().flatten()

df_test['performance_score'] = preds_mean
df_test.to_csv("data/processed/test_with_preds.csv", index=False)
print("Предсказания для теста сохранены в test_with_preds.csv")
