import pandas as pd

def explain(cpu_row):
    """Генерация текстового объяснения для CPU"""
    name = cpu_row.get('cpu_model', 'Неизвестный CPU')
    freq = cpu_row.get('clock_speed', None)
    tdp = cpu_row.get('tdp', None)
    score = cpu_row.get('decision_score', 0)

    # Безопасное извлечение частоты
    try:
        if pd.isna(freq):
            freq_str = "неизвестно"
        else:
            freq_str = f"{round(float(freq), 2)}"
    except (ValueError, TypeError):
        freq_str = "неизвестно"

    # Безопасное извлечение TDP
    try:
        if pd.isna(tdp):
            tdp_str = "н/д"
        else:
            tdp_str = f"{float(tdp):.1f}"
    except (ValueError, TypeError):
        tdp_str = "н/д"

    # Текстовые объяснения по уровню эффективности
    if score >= 0.8:
        summary = "Процессор демонстрирует отличное соотношение производительности и энергопотребления."
    elif score >= 0.4:
        summary = "Процессор имеет сбалансированные характеристики."
    else:
        summary = "Процессор менее эффективен по сравнению с лидерами."

    return (
        f"CPU {name} имеет частоту {freq_str} ГГц и TDP {tdp_str} Вт. "
        f"Итоговый балл по TOPSIS = {score:.3f}. {summary}"
    )
