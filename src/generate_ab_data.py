"""Генератор данных для A/B-теста.

Гипотеза: Кнопка "Купить в 1 клик" увеличивает конверсию из корзины в заказ
с 20% (control) до 22% (treatment) - то есть на 10% относительно.
Остальные шаги воронки не меняются.
"""
import os
import numpy as np
import pandas as pd

N_USERS = 100000
SEED = 42
OUT_DIR = "data"

# Базовые вероятности
P_CLICK = 0.40
P_CART = 0.30
P_ORDER_CONTROL = 0.20
P_ORDER_TREATMENT = 0.22  # +10% относительного прироста
P_PAY = 0.80

def generate_ab_data():
    rng = np.random.default_rng(SEED)
    print(f"Генерация A/B теста для {N_USERS} пользователей...")
    
    user_ids = np.arange(1, N_USERS + 1)
    groups = rng.choice(["control", "treatment"], size=N_USERS)
    
    devices = rng.choice(["iOS", "Android", "Web"], size=N_USERS, p=[0.3, 0.5, 0.2])
    channels = rng.choice(["organic", "paid", "referral"], size=N_USERS, p=[0.4, 0.4, 0.2])
    
    # Воронка
    viewed = np.ones(N_USERS, dtype=int)
    clicked = rng.binomial(1, P_CLICK, size=N_USERS)
    
    added_to_cart = np.zeros(N_USERS, dtype=int)
    cart_mask = clicked == 1
    added_to_cart[cart_mask] = rng.binomial(1, P_CART, size=cart_mask.sum())
    
    # Главное отличие: разная вероятность заказа для групп
    ordered = np.zeros(N_USERS, dtype=int)
    
    # Control
    control_mask = (added_to_cart == 1) & (groups == "control")
    ordered[control_mask] = rng.binomial(1, P_ORDER_CONTROL, size=control_mask.sum())
    
    # Treatment (с эффектом)
    treatment_mask = (added_to_cart == 1) & (groups == "treatment")
    ordered[treatment_mask] = rng.binomial(1, P_ORDER_TREATMENT, size=treatment_mask.sum())
    
    # Оплата
    purchased = np.zeros(N_USERS, dtype=int)
    purchase_mask = ordered == 1
    purchased[purchase_mask] = rng.binomial(1, P_PAY, size=purchase_mask.sum())
    
    # GMV
    gmv = np.zeros(N_USERS)
    paid_mask = purchased == 1
    gmv[paid_mask] = np.round(rng.lognormal(mean=6, sigma=1, size=paid_mask.sum()), 2)
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "group": groups,
        "device": devices,
        "channel": channels,
        "clicked": clicked,
        "added_to_cart": added_to_cart,
        "ordered": ordered,
        "purchased": purchased,
        "gmv": gmv
    })
    
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "ab_test_1.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")

if __name__ == "__main__":
    generate_ab_data()
