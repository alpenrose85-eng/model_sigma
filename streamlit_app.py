import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Sigma-phase temperature inspector", layout="wide")

R = 8.314

COLUMN_RENAMES = {
    "dэкв_мкм": "d_equiv_um",
    "tau_h": "tau_h",
    "T_C": "T_C",
    "G": "G",
    "c_sigma_pct": "c_sigma_pct",
}

SIGMA_TEMP_MIN = 560
SIGMA_TEMP_MAX = 900
SIGMA_TEMP_WIDTH = 8


def sigma_activity(T_K):
    T_C = T_K - 273.15
    lower = 1.0 / (1.0 + np.exp(-(T_C - SIGMA_TEMP_MIN) / SIGMA_TEMP_WIDTH))
    upper = 1.0 / (1.0 + np.exp((T_C - SIGMA_TEMP_MAX) / SIGMA_TEMP_WIDTH))
    return lower * upper


def compute_predicted_diameter(T_K, tau, G, model, m):
    k0 = model["k0"]
    Q = model["Q_J"]
    beta_G = model["beta_G"]
    gamma = sigma_activity(T_K)
    exponent = -Q / (R * T_K)
    return (k0 * gamma * tau * np.exp(beta_G * G) * np.exp(exponent)) ** (1.0 / m)


def solve_temperature_for_growth(model, m, D, tau, G):
    def f(T):
        return compute_predicted_diameter(T, tau, G, model, m) - D

    min_k = SIGMA_TEMP_MIN + 273.15
    max_k = SIGMA_TEMP_MAX + 273.15
    f_min = f(min_k)
    f_max = f(max_k)
    if np.sign(f_min) == np.sign(f_max):
        return None

    a, b = min_k, max_k
    for _ in range(40):
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        if abs(f_mid) < 1e-3:
            return mid
        if np.sign(f_mid) == np.sign(f_min):
            a = mid
            f_min = f_mid
        else:
            b = mid
    return 0.5 * (a + b)


def load_data(uploaded):
    if uploaded is None:
        return pd.DataFrame(), 0, 0

    if uploaded.name.endswith(".xlsx") or uploaded.name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    df = df.rename(columns=COLUMN_RENAMES)
    expected = set(COLUMN_RENAMES.values())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"В файле не хватает колонок: {', '.join(missing)}")

    df = df[list(expected)].copy()
    for col in ["G", "T_C", "tau_h", "d_equiv_um", "c_sigma_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["G", "T_C", "tau_h", "d_equiv_um"])
    df["T_K"] = df["T_C"] + 273.15

    total_rows = len(df)
    mask_valid = (df["G"] > 0) & (df["T_C"] >= SIGMA_TEMP_MIN) & (df["T_C"] <= SIGMA_TEMP_MAX)
    filtered_out = total_rows - mask_valid.sum()
    df = df[mask_valid]
    return df, total_rows, filtered_out


def compute_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rmse_log = math.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))
    return {"rmse": rmse, "r2": r2, "rmse_log": rmse_log}


def compute_temp_metrics(T_true, T_pred):
    mask = np.isfinite(T_pred)
    if not mask.any():
        return {"rmse": float("nan")}
    rmse = math.sqrt(mean_squared_error(T_true[mask], T_pred[mask]))
    return {"rmse": rmse}


def sigma_activity(T_K):
    T_C = T_K - 273.15
    lower = 1.0 / (1.0 + np.exp(-(T_C - SIGMA_TEMP_MIN) / SIGMA_TEMP_WIDTH))
    upper = 1.0 / (1.0 + np.exp((T_C - SIGMA_TEMP_MAX) / SIGMA_TEMP_WIDTH))
    return lower * upper


def compute_predicted_diameter(T_K, tau, G, model, m):
    k0 = model["k0"]
    Q_J = model["Q_J"]
    beta_G = model["beta_G"]
    gamma = sigma_activity(T_K)
    exponent = -Q_J / (R * T_K)
    return (k0 * gamma * tau * np.exp(beta_G * G) * np.exp(exponent)) ** (1.0 / m)


def solve_temperature_for_growth(model, m, D, tau, G):
    def f(T):
        return compute_predicted_diameter(T, tau, G, model, m) - D

    min_k = SIGMA_TEMP_MIN + 273.15
    max_k = SIGMA_TEMP_MAX + 273.15
    f_min = f(min_k)
    f_max = f(max_k)
    if np.sign(f_min) == np.sign(f_max):
        return None

    a, b = min_k, max_k
    for _ in range(40):
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        if abs(f_mid) < 1e-3:
            return mid
        if np.sign(f_mid) == np.sign(f_min):
            a = mid
            f_min = f_mid
        else:
            b = mid
    return 0.5 * (a + b)


def clamp_temperature(T_arr):
    min_K = SIGMA_TEMP_MIN + 273.15
    max_K = SIGMA_TEMP_MAX + 273.15
    return np.where((T_arr >= min_K) & (T_arr <= max_K), T_arr, np.nan)


def fit_growth_model(df, m, include_predictions=False):
    df = df.copy()
    y = m * np.log(df["d_equiv_um"]) - np.log(df["tau_h"])
    X = np.column_stack([
        np.ones(len(df)),
        1.0 / df["T_K"],
        df["G"],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, beta_T, beta_G = beta
    Q_J = -beta_T * R
    k0 = math.exp(intercept)
    d_pred = compute_predicted_diameter(df["T_K"], df["tau_h"], df["G"], {"k0": k0, "beta_G": beta_G, "Q_J": Q_J}, m)
    metrics = compute_metrics(df["d_equiv_um"], d_pred)
    model = {
        "intercept": intercept,
        "beta_T": beta_T,
        "beta_G": beta_G,
        "k0": k0,
        "Q_J": Q_J,
        "Q_kJ_per_mol": Q_J / 1000.0,
        "metric": metrics,
    }
    if include_predictions:
        T_pred = []
        for D_val, tau_val, G_val in zip(df["d_equiv_um"], df["tau_h"], df["G"]):
            T_est = solve_temperature_for_growth(model, m, D_val, tau_val, G_val)
            T_pred.append(np.nan if T_est is None else T_est)
        T_pred = np.array(T_pred, dtype=float)
        mask = ~np.isnan(T_pred)
        temp_rmse = math.sqrt(np.mean((T_pred[mask] - df["T_K"][mask]) ** 2)) if mask.any() else float("nan")
        model.update({
            "D_pred": d_pred,
            "T_pred_K": T_pred,
            "temp_rmse_K": temp_rmse,
        })
    return model


def fit_kG_model(df, include_predictions=False):
    df = df.copy()
    y = np.log(df["d_equiv_um"])
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["tau_h"]),
        np.log(df["G"]),
        1.0 / df["T_K"],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, beta_tau, gamma, beta_T = beta
    y_pred = X @ beta
    d_pred = np.exp(y_pred)
    metrics = compute_metrics(df["d_equiv_um"], d_pred)
    model = {
        "intercept": intercept,
        "beta_tau": beta_tau,
        "gamma": gamma,
        "beta_T": beta_T,
        "metric": metrics,
    }
    if include_predictions:
        value = np.log(df["d_equiv_um"]) - intercept - beta_tau * np.log(df["tau_h"]) - gamma * np.log(df["G"])
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = value / beta_T
            T_pred = np.where(denom > 0, 1.0 / denom, np.nan)
        mask = ~np.isnan(T_pred)
        temp_rmse = math.sqrt(np.mean((T_pred[mask] - df["T_K"][mask]) ** 2)) if mask.any() else float("nan")
        model.update({
            "D_pred": d_pred,
            "y_pred": y_pred,
            "T_pred_K": T_pred,
            "temp_rmse_K": temp_rmse,
        })
    return model


def estimate_temperature_growth(model, D, tau, G, m):
    if model["beta_T"] == 0:
        return None
    value = m * math.log(D) - math.log(tau) - model["intercept"] - model["beta_G"] * G
    denom = value / model["beta_T"]
    if denom <= 0:
        return None
    T = 1.0 / denom
    T = clamp_temperature(np.array([T]))
    return float(T) if np.isfinite(T) else None


def estimate_temperature_kG(model, D, tau, G):
    if model["beta_T"] == 0:
        return None
    value = math.log(D) - model["intercept"] - model["beta_tau"] * math.log(tau) - model["gamma"] * math.log(G)
    denom = value / model["beta_T"]
    if denom <= 0:
        return None
    T = 1.0 / denom
    T = clamp_temperature(np.array([T]))
    return float(T) if np.isfinite(T) else None


def fit_inverse_temp_model(df):
    df = df.copy()
    c_sigma = df["c_sigma_pct"].replace(0, 0.1)
    features = np.column_stack([
        np.log(df["d_equiv_um"]),
        np.log(df["tau_h"]),
        df["G"],
        np.log(c_sigma),
    ])
    model = LinearRegression().fit(features, 1.0 / df["T_K"])
    y_pred = model.predict(features)
    with np.errstate(divide="ignore"):
        T_pred = np.where(y_pred > 0, 1.0 / y_pred, np.nan)
    return {
        "model": model,
        "T_pred_K": T_pred,
        "metrics": compute_temp_metrics(df["T_K"], T_pred),
    }


def fit_boosted_temp_model(df):
    df = df.copy()
    df["c_sigma_pct"] = df["c_sigma_pct"].replace(0, 0.1)
    features = df[["d_equiv_um", "tau_h", "G", "c_sigma_pct"]]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    model.fit(features, df["T_K"])
    T_pred_K = model.predict(features)
    return {
        "model": model,
        "T_pred_K": T_pred_K,
        "metrics": compute_temp_metrics(df["T_K"], T_pred_K),
    }


def main():
    st.title("Sigma-phase temperature model")
    st.markdown(
        f"""
        Анализируем рост σ-фазы в стали 12Х18Н12Т, подбираем параметрические модели и
        получаем оценку температуры эксплуатации по твоим точкам. Загружай свежие CSV/XLSX
        в виджете слева — никаких встроенных данных, только то, что ты проверяешь прямо сейчас.
        σ-фаза стабилизируется начиная примерно {SIGMA_TEMP_MIN}°C и растворяется ближе к {SIGMA_TEMP_MAX}°C,
        поэтому все вычисления выполняются только внутри этого диапазона.
        """
    )

    with st.sidebar.expander("Загрузка данных"):
        uploaded = st.file_uploader("Загрузи CSV или Excel с измерениями", type=["csv", "xlsx", "xls"])
        st.caption("Обязательно: G, T_C, tau_h, dэкв_мкм; c_sigma_pct опционально")

    if uploaded is None:
        st.info("Загрузи файл с новыми экспериментальными точками — после загрузки появятся модели и графики.")
        return

    try:
        df, total_rows, filtered_out = load_data(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return

    if filtered_out > 0:
        st.warning(
            f"{filtered_out} из {total_rows} строк не входят в диапазон температур {SIGMA_TEMP_MIN}–{SIGMA_TEMP_MAX}°C, где σ-фаза стабильна."
        )

    if df.empty:
        st.error("В загруженном файле нет точек внутри допустимого диапазона σ-фазы.")
        return

    st.sidebar.subheader("Выбор роста")
    m_candidates = np.round(np.linspace(1.0, 3.0, 21), 2)
    rmse_by_m = []
    for m_val in m_candidates:
        result = fit_growth_model(df, m_val, include_predictions=False)
        rmse_by_m.append(result["metric"]["rmse"])
    best_m = float(m_candidates[int(np.argmin(rmse_by_m))])
    st.sidebar.text(f"Лучший RMSE в μм → m ≈ {best_m:.2f}")
    selected_m = st.sidebar.slider("Экспонента роста m", 1.0, 3.0, best_m, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Чтобы подставить исторические точки, обнови файл и перезапусти приложение.")

    growth_model = fit_growth_model(df, selected_m, include_predictions=True)
    kG_model = fit_kG_model(df, include_predictions=True)
    inverse_model = fit_inverse_temp_model(df)
    boosted_model = fit_boosted_temp_model(df)

    st.subheader("Параметры моделей")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Ростовая модель")
        st.markdown(
            fr"""
            - $k_0 = e^{{{growth_model['intercept']:.2f}}} \approx {growth_model['k0']:.0f}$
            - $Q = {-growth_model['beta_T'] * R / 1000:.1f}\,\mathrm{{кДж/моль}}$
            - $\beta_G = {growth_model['beta_G']:.3f}$
            - $\mathrm{{RMSE}}(D) = {growth_model['metric']['rmse']:.3f}\,\mu\mathrm{{m}}$
            - $\mathrm{{RMSE}}(\ln D) = {growth_model['metric']['rmse_log']:.3f}$
            - $R^2 = {growth_model['metric']['r2']:.3f}$
            - $\mathrm{{RMSE}}(T) = {growth_model['temp_rmse_K']:.1f}\,\mathrm{{K}}$
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"D^m = k_0 \cdot e^{-Q/(RT)} \cdot \tau \cdot e^{\beta_G G}")

    with col2:
        st.markdown("### Модель с $k_G$(подгон)")
        st.markdown(r"Модель: $\ln D = a + b \ln \tau + \gamma \ln G + \beta_T / T$")
        st.markdown(fr"- $\gamma = {kG_model['gamma']:.3f}$ → $k_G = G^{{{kG_model['gamma']:.3f}}}$")
        st.markdown(
            fr"""
            - $b = {kG_model['beta_tau']:.3f}$
            - $\beta_T = {kG_model['beta_T']:.1f}$
            - $\mathrm{{RMSE}}(D) = {kG_model['metric']['rmse']:.3f}\,\mu\mathrm{{m}}$
            - $\mathrm{{RMSE}}(\ln D) = {kG_model['metric']['rmse_log']:.3f}$
            - $R^2 = {kG_model['metric']['r2']:.3f}$
            - $\mathrm{{RMSE}}(T) = {kG_model['temp_rmse_K']:.1f}\,\mathrm{{K}}$
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Дополнительные модели температуры")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Модель по 1/T")
        st.markdown("$\frac{1}{T} = a + b\ln D + c\ln\tau + dG + e\ln c_{\sigma}$")
        st.markdown(f"- RMSE T: {inverse_model['metrics']['rmse']:.2f}\,K")
    with col4:
        st.markdown("### Градиентный бустинг")
        st.markdown("Функция: $T = f(D, \tau, G, c_{\sigma})$")
        st.markdown(f"- RMSE T: {boosted_model['metrics']['rmse']:.2f}\,K")

    st.subheader("Графики качества")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axs[0, 0].scatter(df["d_equiv_um"], growth_model["D_pred"], c="tab:blue", label="growth")
    axs[0, 0].plot(df["d_equiv_um"], df["d_equiv_um"], linestyle="--", color="gray")
    axs[0, 0].set_xlabel("Наблюдаемый диаметр, μm")
    axs[0, 0].set_ylabel("Предсказанный в ростовой модели")
    axs[0, 0].set_title("Наблюдаемое vs предсказанное — ростовая")

    axs[0, 1].scatter(df["d_equiv_um"], kG_model["D_pred"], c="tab:orange")
    axs[0, 1].plot(df["d_equiv_um"], df["d_equiv_um"], linestyle="--", color="gray")
    axs[0, 1].set_xlabel("Наблюдаемый диаметр, μm")
    axs[0, 1].set_ylabel("Предсказанный в модели k_G")
    axs[0, 1].set_title("Наблюдаемое vs предсказанное — k_G")

    axs[1, 0].hist(df["d_equiv_um"] - growth_model["D_pred"], bins=15, color="tab:blue")
    axs[1, 0].set_title("Остатки — ростовая модель")
    axs[1, 0].set_xlabel("ΔD, μm")

    axs[1, 1].hist(df["d_equiv_um"] - kG_model["D_pred"], bins=15, color="tab:orange")
    axs[1, 1].set_title("Остатки — модель k_G")
    axs[1, 1].set_xlabel("ΔD, μm")

    st.pyplot(fig)

    st.subheader("Температура: предсказания")
    fig_temp, ax_temp = plt.subplots(figsize=(6, 6))
    true_T = df["T_K"]
    for label, preds, color in [
        ("Ростовая", growth_model["T_pred_K"], "tab:blue"),
        ("k_G", kG_model["T_pred_K"], "tab:orange"),
        ("1/T-регрессия", inverse_model["T_pred_K"], "tab:green"),
        ("Boosted", boosted_model["T_pred_K"], "tab:red"),
    ]:
        mask = np.isfinite(preds)
        ax_temp.scatter(true_T[mask], preds[mask], label=label, alpha=0.7, color=color)
    ax_temp.plot(true_T, true_T, linestyle="--", color="gray")
    ax_temp.set_xlabel("Наблюдаемая T, K")
    ax_temp.set_ylabel("Предсказанная T, K")
    ax_temp.set_title("Сравнение температурного прогноза")
    ax_temp.legend()
    st.pyplot(fig_temp)

    st.subheader("Новая точка — оценка температуры")
    with st.form("new_point"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            G_input = st.number_input("G (номер зерна)", min_value=1.0, value=8.0, format="%.1f")
        with col_b:
            tau_input = st.number_input("τ, часы", min_value=1.0, value=4000.0, format="%.1f")
        with col_c:
            d_input = st.number_input("D, мкм", min_value=0.1, value=1.8, format="%.3f")
        submitted = st.form_submit_button("Оценить температуру")
    if submitted:
        T_growth = estimate_temperature_growth(growth_model, d_input, tau_input, G_input, selected_m)
        T_kG = estimate_temperature_kG(kG_model, d_input, tau_input, G_input)
        st.markdown("---")
        T560 = SIGMA_TEMP_MIN + 273.15
        required_d = compute_predicted_diameter(T560, tau_input, G_input, {"k0": growth_model["k0"], "beta_G": growth_model["beta_G"], "Q_J": growth_model["Q_J"]}, selected_m)
        st.write(f"Оценка модели при {SIGMA_TEMP_MIN}°C: $D_{{560}} = {required_d:.3f}\,\mu m$ для заданных G и τ.")
        if T_growth is not None:
            st.write(f"Ростовая модель: $T = {T_growth:.1f}\,K$ ({T_growth - 273.15:.1f} °C)")
        else:
            st.write(
                "Ростовая модель не дала решения → фактический диаметр меньше расчетного при 560°C, значит сигма-фаза ещё не выросла до указанного размера."
            )
        if T_kG is not None:
            st.write(f"Модель с $k_G$: $T = {T_kG:.1f}\,K$ ({T_kG - 273.15:.1f} °C)")
        else:
            st.write(
                "k_G-модель не дала решения → для этой точки σ-фаза оценивается ниже 560°C на основе текущей формулы."
            )

    st.subheader("Данные и предсказания")
    display_df = df[["G", "T_C", "tau_h", "d_equiv_um", "c_sigma_pct", "T_K"]].copy()
    display_df["D_pred_growth"] = growth_model["D_pred"]
    display_df["D_pred_kG"] = kG_model["D_pred"]
    display_df["T_pred_growth_K"] = growth_model["T_pred_K"]
    display_df["T_pred_kG_K"] = kG_model["T_pred_K"]
    display_df["T_pred_inverse_K"] = inverse_model["T_pred_K"]
    display_df["T_pred_boosted_K"] = boosted_model["T_pred_K"]
    display_df["T_pred_growth_C"] = display_df["T_pred_growth_K"] - 273.15
    display_df["T_pred_kG_C"] = display_df["T_pred_kG_K"] - 273.15
    display_df["T_pred_inverse_C"] = display_df["T_pred_inverse_K"] - 273.15
    display_df["T_pred_boosted_C"] = display_df["T_pred_boosted_K"] - 273.15

    def percent_error(actual, predicted):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(predicted) & (predicted != 0), np.abs(actual - predicted) / predicted * 100, np.nan)

    display_df["pct_dev_growth"] = percent_error(display_df["d_equiv_um"], display_df["D_pred_growth"])
    display_df["pct_dev_kG"] = percent_error(display_df["d_equiv_um"], display_df["D_pred_kG"])

    percent_cols = ["pct_dev_growth", "pct_dev_kG"]

    def highlight_large(val):
        if pd.isna(val):
            return ""
        return "background-color: #ffe8e8" if val > 10 else ""

    styled = display_df.style.format(
        {
            "pct_dev_growth": "{:.1f}%",
            "pct_dev_kG": "{:.1f}%",
            "T_pred_growth_C": "{:.1f}",
            "T_pred_kG_C": "{:.1f}",
            "T_pred_inverse_C": "{:.1f}",
            "T_pred_boosted_C": "{:.1f}",
        }
    ).applymap(highlight_large, subset=percent_cols)

    st.dataframe(styled)
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать таблицу с предсказаниями", csv, "predictions.csv", "text/csv")
    st.markdown("В таблице указаны температуры в °C и процентное отклонение экспериментального диаметра от предсказанного (выделено, если >10%).")

    df_analysis = display_df.copy()
    df_analysis["error_growth"] = df_analysis["d_equiv_um"] - df_analysis["D_pred_growth"]
    df_analysis["abs_pct_growth"] = percent_error(df_analysis["d_equiv_um"], df_analysis["D_pred_growth"])

    summary_by_G = (
        df_analysis.groupby("G")
        .agg(
            count=("d_equiv_um", "count"),
            mean_abs_pct=("abs_pct_growth", "mean"),
            error_mean=("error_growth", "mean"),
            error_std=("error_growth", "std"),
        )
        .sort_values("mean_abs_pct")
    )

    df_analysis["temp_bin"] = pd.cut(df_analysis["T_C"], bins=[-1, 600, 650, 700, 750, 1000], labels=["≤600", "600-650", "650-700", "700-750", ">750"])
    summary_by_temp = (
        df_analysis.groupby("temp_bin")
        .agg(
            count=("d_equiv_um", "count"),
            mean_abs_pct=("abs_pct_growth", "mean"),
            error_std=("error_growth", "std"),
        )
        .sort_index()
    )

    top_outliers = df_analysis.sort_values("abs_pct_growth", ascending=False).head(5)[
        ["G", "T_C", "tau_h", "d_equiv_um", "D_pred_growth", "abs_pct_growth"]
    ]

    st.subheader("Анализ точности модели")
    fig_bins, ax_bins = plt.subplots(figsize=(6, 4))
    summary_by_temp_plot = summary_by_temp.reset_index()
    ax_bins.bar(summary_by_temp_plot['temp_bin'].astype(str), summary_by_temp_plot['mean_abs_pct'], color='tab:blue', alpha=0.7)
    ax_bins.set_xlabel("Температурный диапазон, °C")
    ax_bins.set_ylabel("Среднее % отклонение")
    ax_bins.set_title("Точность по температурным диапазонам")
    ax_bins.grid(axis='y', linestyle='--', alpha=0.5)

    df_analysis['tau_bin'] = pd.cut(df_analysis['tau_h'], bins=[0, 2000, 5000, 10000, 20000], labels=["≤2000", "2000-5000", "5000-10000", ">10000"])
    summary_by_tau = (
        df_analysis.groupby('tau_bin')
        .agg(count=('d_equiv_um', 'count'), mean_abs_pct=('abs_pct_growth', 'mean'))
        .sort_index()
    )
    fig_tau, ax_tau = plt.subplots(figsize=(6, 4))
    summary_by_tau_plot = summary_by_tau.reset_index()
    ax_tau.plot(summary_by_tau_plot['tau_bin'].astype(str), summary_by_tau_plot['mean_abs_pct'], marker='o', color='tab:green')
    ax_tau.set_xlabel("Наработка, ч")
    ax_tau.set_ylabel("Среднее % отклонение")
    ax_tau.set_title("Точность по наработке")
    ax_tau.grid(True, linestyle='--', alpha=0.5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### По номеру зерна")
        st.write("Чем меньше среднее абсолютное отклонение, тем стабильнее модель описывает зерно.")
        st.dataframe(summary_by_G.style.format({"mean_abs_pct": "{:.1f}%", "error_mean": "{:.3f}", "error_std": "{:.3f}"}))
    with col2:
        st.markdown("### По температурным диапазонам")
        st.write("Наилучшая точность заметна в диапазоне 600-650°C, экстремальные значения дают больше разброса.")
        st.pyplot(fig_bins)

    st.subheader("Анализ по наработке")
    st.write("Рассчитано на основе бинов наработки: модель стабильно работает до ≈5000 ч, дальше ошибка растёт.")
    st.pyplot(fig_tau)
    st.dataframe(summary_by_tau.style.format({"mean_abs_pct": "{:.1f}%", "count": "{:.0f}"}))

    st.markdown("### Топ-5 точек с наибольшим отклонением")
    st.dataframe(top_outliers.style.format({"abs_pct_growth": "{:.1f}%"}))

    st.markdown("---")
    st.info(
        "– `m` лучше выбирать по графику RMSE (в боковой панели).\n"
        "– Загружай новые точки через uploader — приложение всегда работает с тем файлом, который ты сейчас проверяешь."
    )


if __name__ == "__main__":
    main()
