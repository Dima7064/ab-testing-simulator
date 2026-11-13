"""Генератор синтетических данных для e-commerce A/B тестов.

Мы создаем базовую воронку без каких-либо эффектов.
Все пользователи случайно распределяются на Control и Treatment.
"""
import os
import numpy as np
import pandas as pd

# Настройки
N_USERS = 100000
SEED = 42
OUT_DIR = "data"

# Базовые вероятности (Baseline) на каждом шаге воронки
P_VIEW = 1.0       # 100% заходят на сайт
P_CLICK = 0.40     # 40% кликают по товару
P_CART = 0.30      # 30% из кликнувших добавляют в корзину
P_ORDER = 0.20     # 20% из добавивших оформляют заказ
P_PAY = 0.80       # 80% из оформивших оплачивают

def generate_ecommerce_data():
    rng = np.random.default_rng(SEED)
    
    print(f"Генерация {N_USERS} пользователей...")
    
    # 1. Создаем пользователей
    user_ids = np.arange(1, N_USERS + 1)
    
    # 2. Рандомизация (50/50)
    groups = rng.choice(["control", "treatment"], size=N_USERS)
    
    # 3. Характеристики пользователей (для сегментации)
    devices = rng.choice(["iOS", "Android", "Web"], size=N_USERS, p=[0.3, 0.5, 0.2])
    channels = rng.choice(["organic", "paid", "referral"], size=N_USERS, p=[0.4, 0.4, 0.2])
    
    # 4. Генерация воронки
    # Просмотры
    viewed = np.ones(N_USERS, dtype=int)
    
    # Клики
    clicked = rng.binomial(1, P_CLICK, size=N_USERS)
    
    # Корзина (только если был клик)
    added_to_cart = np.zeros(N_USERS, dtype=int)
    cart_mask = clicked == 1
    added_to_cart[cart_mask] = rng.binomial(1, P_CART, size=cart_mask.sum())
    
    # Заказ (только если была корзина)
    ordered = np.zeros(N_USERS, dtype=int)
    order_mask = added_to_cart == 1
    ordered[order_mask] = rng.binomial(1, P_ORDER, size=order_mask.sum())
    
    # Оплата (только если был заказ)
    purchased = np.zeros(N_USERS, dtype=int)
    purchase_mask = ordered == 1
    purchased[purchase_mask] = rng.binomial(1, P_PAY, size=purchase_mask.sum())
    
    # 5. GMV (выручка) для тех, кто оплатил (случайные чеки)
    gmv = np.zeros(N_USERS)
    paid_mask = purchased == 1
    # Лог-нормальное распределение для чеков (большинство мелких, мало крупных)
    gmv[paid_mask] = np.round(rng.lognormal(mean=6, sigma=1, size=paid_mask.sum()), 2)
    
    # 6. Собираем DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "group": groups,
        "device": devices,
        "channel": channels,
        "viewed": viewed,
        "clicked": clicked,
        "added_to_cart": added_to_cart,
        "ordered": ordered,
        "purchased": purchased,
        "gmv": gmv
    })
    
    # Сохраняем
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, "baseline_data.csv")
    df.to_csv(filepath, index=False)
    print(f"Данные сохранены в {filepath}")
    print(f"Средняя конверсия в покупку: {df['purchased'].mean():.2%}")
    print(f"Средний GMV на пользователя (ARPU): {df['gmv'].mean():.2f} руб.")

if __name__ == "__main__":
    generate_ecommerce_data()
