import numpy as np
import pandas as pd

def generate_synthetic(df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    if n_samples is None:
        n_samples = len(df)

    synth = df.sample(n_samples, replace=True).reset_index(drop=True)

    # шум на числовые признаки
    num_cols = synth.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        noise = np.random.uniform(0.85, 1.15, len(synth))
        synth[col] = synth[col].astype(float) * noise

    # категориальные — перемешиваем
    cat_cols = synth.select_dtypes(include=['object']).columns
    for col in cat_cols:
        synth[col] = np.random.choice(df[col].dropna().unique(), len(synth))

    # уникальные имена CPU моделей
    if 'SVR.CPU.CPU Model' in df.columns:
        synth['SVR.CPU.CPU Model'] = (
            synth['SVR.CPU.CPU Model']
            + "_synthetic_"
            + np.random.randint(1000, 999999, len(synth)).astype(str)
        )

    # перегенерируем производные признаки (если есть)
    if {'RESULT.Value', 'SVR.Power.TDP', 'SVR.CPU.Base Frequency'}.issubset(synth.columns):
        synth['perf_per_watt'] = synth['RESULT.Value'] / synth['SVR.Power.TDP']
        synth['perf_per_ghz'] = synth['RESULT.Value'] / synth['SVR.CPU.Base Frequency']

        if 'SVR.Memory.MemTotal' in synth.columns:
            synth['mem_efficiency'] = synth['RESULT.Value'] / (synth['SVR.Memory.MemTotal'] + 1e-6)

    return synth
