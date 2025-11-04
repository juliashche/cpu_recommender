import pandas as pd
from module.recommender.hybrid_recommender import hybrid
from module.decision.decision_agent import topsis, dsm_hypothesis

# Загрузка готовых данных с performance_score
data_path = "C:/Users/Jull/PycharmProjects/rec_sys/NCPP/data/processed/test_with_preds.csv"
df = pd.read_csv(data_path)


# Переименуем колонки для совместимости с hybrid_recommender
df.rename(columns={
    'SVR.CPU.CPU Model': 'cpu_model',
    'SVR.CPU.Base Frequency': 'clock_speed',
    'SVR.Power.TDP': 'tdp',
    'RESULT.Value': 'performance_score'
}, inplace=True)

target_model = "Intel (R) Xeon (R) CPU Max 9460"

# Построение гибридных рекомендаций (content-based + performance_score)
df['tabm_score'] = (df['performance_score'] - df['performance_score'].min()) / \
                   (df['performance_score'].max() - df['performance_score'].min() + 1e-9)

# content-based score
from module.recommender.hybrid_recommender import content_based
content = content_based(df, target_model, top_k=10)  # больше топ для дальнейшего ранжирования

# Объединяем
merged = content.merge(df[['cpu_model', 'tabm_score']], on='cpu_model', how='left').fillna(0)
merged['hybrid_score'] = 0.5 * merged['content_score'] + 0.5 * merged['tabm_score']

top5 = merged.sort_values('hybrid_score', ascending=False).head(5)

# Многокритериальное ранжирование TOPSIS
decision = topsis(top5)

# Вывод финальных рекомендаций и гипотезы DSM
print("\nФинальные рекомендации:")
for _, row in decision.iterrows():
    print(f"{row['cpu_model']} (Score={row['decision_score']:.3f})")
    if 'clock_speed' in row:
        print(f"  Частота: {row['clock_speed']:.2f} ГГц, TDP: {row['tdp']:.1f} Вт")

print("\nГипотеза DSM:")
print(dsm_hypothesis(decision))
