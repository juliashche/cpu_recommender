import numpy as np
import pandas as pd

def topsis(df):
    """
    TOPSIS для выбора оптимального CPU:
    - performance_score (maximize)
    - tdp (minimize)
    """
    df = df.copy()

    cols = ['performance_score', 'tdp']
    if not all(col in df.columns for col in cols):
        print("Недостаточно данных для TOPSIS")
        return pd.DataFrame()

    # Нормализация
    norm = df[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))

    # Определяем лучшее и худшее решения
    best = [norm['performance_score'].max(), norm['tdp'].min()]
    worst = [norm['performance_score'].min(), norm['tdp'].max()]

    # Расчёт расстояний
    dist_best = np.sqrt(((norm - best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((norm - worst) ** 2).sum(axis=1))

    # Итоговый TOPSIS score
    df['decision_score'] = dist_worst / (dist_best + dist_worst + 1e-9)
    df = df.sort_values('decision_score', ascending=False)

    # Безопасное возвращение: clock_speed может отсутствовать
    cols_to_return = ['cpu_model', 'performance_score', 'tdp', 'decision_score']
    if 'clock_speed' in df.columns:
        cols_to_return.insert(3, 'clock_speed')

    return df[cols_to_return]


def dsm_hypothesis(df):
    """
    ДСМ-агент (Decision Support Model).
    Формирует гипотезу о том, какие характеристики влияют на эффективность CPU.
    """
    if 'performance_score' not in df.columns or 'tdp' not in df.columns:
        return "Недостаточно данных для анализа."

    # Простая логика: ищем зависимость эффективности от TDP и частоты
    avg_eff = (df['performance_score'] / df['tdp']).mean()
    high_eff = df[df['performance_score'] / df['tdp'] > avg_eff]

    if high_eff.empty:
        return "Нет процессоров, превосходящих среднюю эффективность."

    mean_clock = high_eff['clock_speed'].mean() if 'clock_speed' in df.columns else None
    hypothesis = "Высокая эффективность чаще встречается у CPU "
    if mean_clock:
        hypothesis += f"с частотой около {mean_clock:.2f} ГГц "
    hypothesis += f"и TDP ниже среднего ({df['tdp'].mean():.1f} Вт)."

    return hypothesis
