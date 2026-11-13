import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="A/B Testing Master", layout="wide", page_icon="🧪")

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("⚙️ Параметры эксперимента")
st.sidebar.markdown("Двигай ползунки — всё пересчитывается **мгновенно**.")

metric_type = st.sidebar.radio(
    "Выбери метрику:",
    ["Конверсия (CR)", "Средний чек (ARPU)", "Удержание (Retention D7)"],
    help="Разные метрики требуют разных статистических тестов (Z-test для конверсий, T-test для средних)."
)

if metric_type == "Конверсия (CR)":
    baseline = st.sidebar.slider("Базовая конверсия, %", 1.0, 30.0, 10.0, 0.5) / 100.0
elif metric_type == "Средний чек (ARPU)":
    baseline = st.sidebar.slider("Базовый средний чек, руб.", 100, 5000, 1000, 100)
else:
    baseline = st.sidebar.slider("Базовый Retention, %", 5.0, 60.0, 20.0, 1.0) / 100.0

mde = st.sidebar.slider("Ожидаемый прирост (MDE), %", 1, 50, 10, 1) / 100.0
n_users = st.sidebar.number_input("Размер выборки на группу", 1000, 200000, 20000, 5000)

st.sidebar.markdown("---")
st.sidebar.subheader("🕳 Искусственные поломки")
srm_shift = st.sidebar.slider("Сдвиг размеров групп (SRM), чел.", 0, 5000, 0, 100, help="0 = группы равны. 1000 = в Treatment на 1000 больше.")
apply_cuped_flag = st.sidebar.checkbox("Применить CUPED", value=False, help="Снижает дисперсию, используя данные 'до' эксперимента.")

# --- СИМУЛЯЦИЯ ---
@st.cache_data
def get_data(n, base, mde_val, srm, metric, cuped):
    rng = np.random.default_rng(42)
    n_c = n
    n_t = n + srm
    
    if "Конверсия" in metric:
        c_data = rng.binomial(1, base, size=n_c)
        t_data = rng.binomial(1, base * (1 + mde_val), size=n_t)
        return c_data, t_data
    
    elif "ARPU" in metric:
        c_data = rng.lognormal(np.log(base), 0.5, size=n_c)
        t_data = rng.lognormal(np.log(base * (1 + mde_val)), 0.5, size=n_t)
        if cuped:
            # Симуляция CUPED
            pre_c = c_data * 0.8 + rng.normal(0, 100, n_c)
            pre_t = t_data * 0.8 + rng.normal(0, 100, n_t)
            theta = 0.6 # Искусственная корреляция
            c_data = c_data - theta * (pre_c - pre_c.mean())
            t_data = t_data - theta * (pre_t - pre_t.mean())
        return c_data, t_data
    
    else:
        c_data = rng.binomial(1, base, size=n_c)
        t_data = rng.binomial(1, base * (1 + mde_val), size=n_t)
        return c_data, t_data

c_data, t_data = get_data(n_users, baseline, mde, srm_shift, metric_type, apply_cuped_flag)

# Статистика
if "ARPU" in metric_type:
    mean_c, mean_t = c_data.mean(), t_data.mean()
    t_stat, p_val = stats.ttest_ind(t_data, c_data, equal_var=False)
    se_c, se_t = c_data.std()/np.sqrt(len(c_data)), t_data.std()/np.sqrt(len(t_data))
    test_name = "T-test (средние)"
    fmt = "{:.2f} ₽"
else:
    mean_c, mean_t = c_data.mean(), t_data.mean()
    z_stat, p_val = proportions_ztest([t_data.sum(), c_data.sum()], [len(t_data), len(c_data)])
    se_c, se_t = np.sqrt(mean_c*(1-mean_c)/len(c_data)), np.sqrt(mean_t*(1-mean_t)/len(t_data))
    test_name = "Z-test (пропорции)"
    fmt = "{:.2%}"

ci_c = (mean_c - 1.96*se_c, mean_c + 1.96*se_c)
ci_t = (mean_t - 1.96*se_t, mean_t + 1.96*se_t)

# --- ГЛАВНЫЙ ЭКРАН ---
st.title("🧪 A/B Testing Masterclass")
tabs = st.tabs(["📊 Симулятор", "🧠 Продвинутая математика", "📖 Словарь метрик"])

# --- ВКЛАДКА 1: СИМУЛЯТОР ---
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Control", fmt.format(mean_c), f"95% CI: [{fmt.format(ci_c[0])}, {fmt.format(ci_c[1])}]")
    col2.metric("Treatment", fmt.format(mean_t), f"95% CI: [{fmt.format(ci_t[0])}, {fmt.format(ci_t[1])}]")
    
    # Вердикт
    st.markdown("### 📝 Вердикт")
    if srm_shift > 0:
        st.error("⚠️ **SRM DETECTED!** Размеры групп не равны (сдвиг = {}). Результаты теста **невалидны**. Если ты видишь SRM, ты должен остановить анализ и идти к разработчикам чинить логгер.".format(srm_shift))
    elif p_val < 0.05:
        st.success(f"✅ **Катим!** p-value ({p_val:.4f}) < 0.05. Эффект статистически значим.")
    else:
        st.warning(f"❌ **Не катим.** p-value ({p_val:.4f}) > 0.05. Нет доказательств эффекта.")
        
    st.info(f"Использован тест: **{test_name}** | p-value = **{p_val:.4f}**")

    # График с CI
    st.markdown("### График конверсий с доверительными интервалами (CI)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Control", "Treatment"], [mean_c, mean_t], yerr=[1.96*se_c, 1.96*se_t], capsize=10, color=['steelblue', 'coral'])
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение групп с 95% CI (усы = погрешность)")
    st.pyplot(fig)

# --- ВКЛАДКА 2: ПРОДВИНУТАЯ МАТЕМАТИКА ---
with tabs[1]:
    st.header("1. CUPED (Снижение дисперсии)")
    st.markdown("CUPED использует данные *до* эксперимента, чтобы сузить колокол распределения. Узкий колокол = легче найти эффект.")
    
    if apply_cuped_flag:
        fig_cuped, ax_cuped = plt.subplots(figsize=(10, 4))
        # Бутстрэп для визуализации
        boot_c = [np.random.choice(c_data, 1000).mean() for _ in range(300)]
        boot_t = [np.random.choice(t_data, 1000).mean() for _ in range(300)]
        sns.kdeplot(boot_c, label="Control (CUPED)", fill=True, ax=ax_cuped, color="blue")
        sns.kdeplot(boot_t, label="Treatment (CUPED)", fill=True, ax=ax_cuped, color="orange")
        ax_cuped.set_title("Распределение средних С ПРИМЕНЕНИЕМ CUPED (колоколы узкие)")
        ax_cuped.legend()
        st.pyplot(fig_cuped)
    else:
        st.warning("👈 Включи галочку **'Применить CUPED'** в боковой панели, чтобы увидеть эффект.")
        
    st.markdown("---")
    st.header("2. Multiple Testing (Множественные проверки)")
    st.markdown("Если проверить 10 метрик без эффекта, 1 из них случайно покажет p < 0.05. Это ложная тревога (Ошибка I рода). Нажми кнопку ниже, чтобы сгенерировать 10 метрик.")
    
    if st.button("Сгенерировать 10 метрик (без эффекта)"):
        p_vals = [stats.ttest_ind(np.random.normal(50, 10, 1000), np.random.normal(50, 10, 1000))[1] for _ in range(10)]
        fig_mt, ax_mt = plt.subplots(figsize=(10, 3))
        colors = ['green' if p < 0.05 else 'gray' for p in p_vals]
        ax_mt.bar(range(1, 11), p_vals, color=colors)
        ax_mt.axhline(y=0.05, color="red", linestyle="--", label="Порог 0.05")
        ax_mt.set_xticks(range(1, 11))
        ax_mt.set_xlabel("Номер метрики")
        ax_mt.set_ylabel("p-value")
        ax_mt.legend()
        st.pyplot(fig_mt)
        
        false_positives = sum(1 for p in p_vals if p < 0.05)
        if false_positives > 0:
            st.error(f"⚠️ Найдено {false_positives} 'значимых' метрик! Но мы-то знаем, что эффекта НЕТ ни в одной. Это и есть Multiple Testing. В реальности это лечится поправкой Бонферрони (порог 0.05 / 10 = 0.005).")

# --- ВКЛАДКА 3: СЛОВАРЬ ---
with tabs[2]:
    st.header("📖 Шпаргалка продуктового аналитика")
    
    with st.expander("Базовые продуктовые метрики"):
        st.write("""
        * **DAU / MAU** — Daily/Monthly Active Users. Сколько уникальных людей заходят в день/месяц.
        * **Sticky Factor** — DAU / MAU. Показывает, насколько часто люди возвращаются (для банка норм 20-40%).
        * **Retention (Удержание)** — % людей, вернувшихся через N дней (D1, D7, D30).
        * **Churn (Отток)** — % людей, которые перестали пользоваться продуктом.
        * **ARPU** — Average Revenue Per User. Средняя выручка на одного пользователя.
        * **LTV** — Lifetime Value. Сколько денег принесет один клиент за всю жизнь.
        * **CAC** — Customer Acquisition Cost. Сколько денег мы потратили на привлечение.
        * **Юнит-экономика** — правило LTV / CAC > 3. Если меньше — бизнес убыточен.
        """)
        
    with st.expander("Статистические термины (A/B)"):
        st.write("""
        * **Alpha (0.05)** — риск Ошибки I рода (ложноположительный). 5% случаев мы скажем, что эффект есть, когда его нет.
        * **Power (0.8)** — вероятность найти эффект, если он есть (1 - Ошибка II рода).
        * **p-value** — вероятность увидеть такую разницу случайно. Если < 0.05 — эффект значим.
        * **CI (Доверительный интервал)** — диапазон, в котором с вероятностью 95% находится истинный эффект. Если CI не включает 0 — эффект значим.
        * **MDE** — минимальный прирост, который мы хотим задетектировать.
        * **SRM** — поломка рандомизации (группы разного размера).
        """)