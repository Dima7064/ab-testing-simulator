"""Генератор данных для демонстрации CUPED.

У нас есть метрика "GMV пользователя" (Y).
И есть историческая метрика "GMV в прошлом месяце" (X).
X и Y коррелируют.
"""
import os
import numpy as np
import pandas as pd

N_USERS = 50000
SEED = 123
OUT_DIR = "data"

def generate_cuped():
    rng = np.random.default_rng(SEED)
    print(f"Генерация {N_USERS} пользователей для CUPED...")
    
    user_ids = np.arange(1, N_USERS + 1)
    groups = rng.choice(["control", "treatment"], size=N_USERS)
    
    # X - предэкспериментальный GMV (прошлый месяц)
    # Делаем лог-нормальное распределение, чтобы имитировать траты
    pre_gmv = rng.lognormal(mean=5, sigma=1, size=N_USERS)
    
    # Y - GMV в эксперименте (текущий месяц)
    # Он зависит от pre_gmv (корреляция) + случайный шум
    # Добавим эффект +5% для Treatment
    effect = np.where(groups == "treatment", 1.05, 1.0)
    post_gmv = pre_gmv * 0.8 * effect + rng.lognormal(mean=4, sigma=1, size=N_USERS)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "group": groups,
        "pre_gmv": pre_gmv,
        "post_gmv": post_gmv
    })
    
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "cuped_data.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generate_cuped()
