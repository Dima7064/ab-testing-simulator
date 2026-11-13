"""Генератор данных для демонстрации Multiple Testing (p-hacking).

Эффекта НЕТ ни в одной метрике. Все различия - чистый случайный шум.
Но из-за того, что метрик много, мы найдем "значимые" различия.
"""
import os
import numpy as np
import pandas as pd

N_USERS = 20000
SEED = 777
OUT_DIR = "data"

def generate_multiple_testing():
    rng = np.random.default_rng(SEED)
    print(f"Генерация {N_USERS} пользователей (эффекта нет нигде)...")
    
    user_ids = np.arange(1, N_USERS + 1)
    groups = rng.choice(["control", "treatment"], size=N_USERS)
    
    # Создаем 10 случайных метрик, никак не связанных с группой
    df = pd.DataFrame({"user_id": user_ids, "group": groups})
    
    for i in range(1, 11):
        # Это непрерывные метрики (время в секундах, клики и т.д.)
        # Генерируем нормальное распределение. Группы НЕ отличаются.
        df[f"metric_{i}"] = rng.normal(loc=50, scale=15, size=N_USERS)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "multiple_testing.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generate_multiple_testing()
