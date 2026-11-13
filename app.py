"""Интерактивный дашборд для симуляции A/B-тестов в E-commerce."""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(page_title="A/B Testing Playground", layout="wide", page_icon="🧪")

st.title("🧪 Симулятор A/B-теста: NeoBank E-commerce")
st.markdown("""
Этот инструмент позволяет смоделировать A/B-тест на синтетических данных и посмотреть, 
как размер выборки и реальный эффект влияют на **p-value** и **статистическую мощность**.
""")

# --- БОКОВАЯ ПАНЕЛЬ (ВВОДНЫЕ ДАННЫЕ) ---
st.sidebar.header("Параметры эксперимента")

baseline_cr = st.sidebar.slider(
    "Базовая конверсия (Control), %", 
    min_value=1.0, max_value=20.0, value=10.0, step=0.5
) / 100.0

expected_lift = st.sidebar.slider(
    "Ожидаемый относительный прирост (MDE), %", 
    min_value=1, max_value=50, value=10, step=1
) / 100.0

n_users_per_group = st.sidebar.number_input(
    "Размер выборки в каждой группе", 
    min_value=1000, max_value=500000, value=20000, step=5000
)

alpha = st.sidebar.slider(
    "Уровень значимости (Alpha)", 
    min_value=0.01, max_value=0.10, value=0.05, step=0.01
)

power_target = st.sidebar.slider(
    "Целевая мощность (Power)", 
    min_value=0.50, max_value=0.99, value=0.80, step=0.05
)

# --- ВЫЧИСЛЕНИЯ ---
# 1. Расчет MDE и требуемого размера выборки
# Эффект Коэна (Cohen's h) для пропорций
def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

treatment_cr = baseline_cr * (1 + expected_lift)
effect_size = cohens_h(treatment_cr, baseline_cr)

power_analysis = NormalIndPower()
required_n = power_analysis.solve_power(
    effect_size=effect_size, alpha=alpha, power=power_target, ratio=1.0
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Требуемая выборка")
st.sidebar.metric(
    label="Нужно пользователей на группу", 
    value=f"{int(required_n):,}",
    delta=f"У тебя: {n_users_per_group:,}",
    delta_color="normal" if n_users_per_group >= required_n else "inverse"
)

if n_users_per_group < required_n:
    st.sidebar.warning("⚠️ Выборки меньше требуемой. Высок риск ошибки II рода (пропустить эффект).")

# --- ГЛАВНАЯ ЧАСТЬ ---
if st.button("🚀 Запустить симуляцию A/B-теста", type="primary"):
    # Генерируем данные
    rng = np.random.default_rng()  # каждый раз новый сид для реализма
    
    # Control
    c_conv = rng.binomial(1, baseline_cr, size=n_users_per_group)
    # Treatment
    t_conv = rng.binomial(1, treatment_cr, size=n_users_per_group)
    
    # Считаем конверсии
    cr_c = c_conv.mean()
    cr_t = t_conv.mean()
    
    # Z-test
    z_stat, p_val = proportions_ztest(
        [t_conv.sum(), c_conv.sum()], 
        [n_users_per_group, n_users_per_group]
    )
    
    # Доверительные интервалы (упрощенные)
    se = np.sqrt(cr_c * (1 - cr_c) / n_users_per_group)
    ci_c = (cr_c - 1.96 * se, cr_c + 1.96 * se)
    ci_t = (cr_t - 1.96 * se, cr_t + 1.96 * se)

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Конверсия Control", f"{cr_c:.2%}", f"CI: [{ci_c[0]:.2%}, {ci_c[1]:.2%}]")
    col2.metric("Конверсия Treatment", f"{cr_t:.2%}", f"CI: [{ci_t[0]:.2%}, {ci_t[1]:.2%}]")
    col3.metric("p-value", f"{p_val:.4f}")

    # Вердикт
    st.markdown("### 📝 Вердикт")
    if p_val < alpha:
        st.success(f"✅ **Статистически значимо!** (p < {alpha}). Фича работает. Можно катить.")
    else:
        st.error(f"❌ **Статистически не значимо.** (p = {p_val:.4f} > {alpha}). Мы не можем утверждать, что фича работает. Рекомендуется продлить тест или не катить фичу.")

    # Визуализация
    st.markdown("### 📈 Визуализация распределений (Bootstrap)")
    
    # Бутстрэп для визуализации плотности
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Генерируем много средних (имитация ЦПТ)
    boot_c = [rng.choice(c_conv, size=n_users_per_group, replace=True).mean() for _ in range(500)]
    boot_t = [rng.choice(t_conv, size=n_users_per_group, replace=True).mean() for _ in range(500)]
    
    sns.kdeplot(boot_c, label="Control", fill=True, ax=ax, color="blue")
    sns.kdeplot(boot_t, label="Treatment", fill=True, ax=ax, color="orange")
    ax.set_title("Распределение средних конверсий (Bootstrap)")
    ax.set_xlabel("Conversion Rate")
    ax.legend()
    
    st.pyplot(fig)
