"""Генератор данных для демонстрации Peeking и Novelty Effect."""
import os
import numpy as np
import pandas as pd

N_USERS_PER_DAY = 20000
DAYS = 14
SEED = 99
OUT_DIR = "data"

def generate_novelty_data():
    rng = np.random.default_rng(SEED)
    print(f"Генерация данных за {DAYS} дней...")
    
    records = []
    
    for day in range(1, DAYS + 1):
        # В первые 3 дня Treatment имеет огромный прирост (+20%)
        # С 4-го дня эффект пропадает (0%), а к концу теста становится отрицательным (-2%)
        if day <= 3:
            p_treatment = 0.12
        elif day <= 7:
            p_treatment = 0.10  # базовый уровень
        else:
            p_treatment = 0.09  # стало хуже
            
        p_control = 0.10  # Конверсия всегда 10%
        
        # Control
        c_conv = rng.binomial(1, p_control, size=N_USERS_PER_DAY)
        for val in c_conv:
            records.append({"day": day, "group": "control", "purchased": val})
            
        # Treatment
        t_conv = rng.binomial(1, p_treatment, size=N_USERS_PER_DAY)
        for val in t_conv:
            records.append({"day": day, "group": "treatment", "purchased": val})
            
    df = pd.DataFrame(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "novelty_peeking.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generate_novelty_data()
