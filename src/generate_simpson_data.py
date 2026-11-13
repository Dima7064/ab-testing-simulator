"""Генератор данных для демонстрации Парадокса Симпсона.

Сценарий:
- У нас есть две платформы: iOS и Android.
- На iOS базовая конверсия высокая (10%), на Android низкая (2%).
- Новая фича (Treatment) на самом деле УЛУЧШАЕТ конверсию на ОБЕИХ платформах на 20% относительно.
- НО! В группу Treatment случайно попали 80% пользователей iOS и только 20% Android.
- А в Control попали 20% iOS и 80% Android.
- В результате общая конверсия Treatment покажется ниже, чем в Control.
"""
import os
import numpy as np
import pandas as pd

N_USERS = 100000
SEED = 2024  # другой сид, чтобы данные были другими
OUT_DIR = "data"

def generate_simpson_data():
    rng = np.random.default_rng(SEED)
    print(f"Генерация данных с Парадоксом Симпсона ({N_USERS} пользователей)...")
    
    # Создаем пользователей
    user_ids = np.arange(1, N_USERS + 1)
    
    # --- ВОТ ЗДЕСЬ ЗАШИТА ОШИБКА ---
    # Мы НЕ делаем случайное распределение. Мы искусственно смещаем группы.
    # В реальности это может случиться, если логгер событий срабатывает по-разному на iOS и Android.
    
    # Создаем список групп с перекосом
    groups = np.array(["treatment"] * N_USERS)
    groups[:20000] = "control"  # 20 000 в Control, 80 000 в Treatment
    
    # Платформы
    devices = np.zeros(N_USERS, dtype=object)
    
    # В Control: 20% iOS (4000) и 80% Android (16000)
    devices[:20000] = rng.choice(["iOS", "Android"], size=20000, p=[0.2, 0.8])
    # В Treatment: 80% iOS (64000) и 20% Android (16000)
    devices[20000:] = rng.choice(["iOS", "Android"], size=80000, p=[0.8, 0.2])
    
    # Вероятности конверсии
    # Базовая конверсия: iOS = 10%, Android = 2%
    # Эффект фичи: +20% относительно (то есть 12% и 2.4%)
    
    # Control
    p_conv_control = np.where(devices[:20000] == "iOS", 0.10, 0.02)
    # Treatment (фича работает лучше!)
    p_conv_treatment = np.where(devices[20000:] == "iOS", 0.12, 0.024)
    
    # Генерируем конверсии
    purchased = np.zeros(N_USERS, dtype=int)
    purchased[:20000] = rng.binomial(1, p_conv_control, size=20000)
    purchased[20000:] = rng.binomial(1, p_conv_treatment, size=80000)
    
    # GMV для тех, кто купил
    gmv = np.zeros(N_USERS)
    paid_mask = purchased == 1
    gmv[paid_mask] = np.round(rng.lognormal(mean=6, sigma=0.5, size=paid_mask.sum()), 2)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "group": groups,
        "device": devices,
        "purchased": purchased,
        "gmv": gmv
    })
    
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "simpson_paradox.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generate_simpson_data()

