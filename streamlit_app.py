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


def load_data(uploaded):
    if uploaded is None:
        return pd.DataFrame()

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
    df = df[df["G"] > 0]
    return df


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
    y_pred = X @ beta
    d_pred = np.exp((y_pred + np.log(df["tau_h"])) / m)
    metrics = compute_metrics(df["d_equiv_um"], d_pred)
    model = {
        "intercept": intercept,
        "beta_T": beta_T,
        "beta_G": beta_G,
        "k0": math.exp(intercept),
        "Q_kJ_per_mol": -beta_T * R / 1000.0,
        "metric": metrics,
    }
    if include_predictions:
        value = m * np.log(df["d_equiv_um"]) - np.log(df["tau_h"]) - intercept - beta_G * df["G"]
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


def fit_kG_model(df, include_predictions=False):
    df = df.copy()
    df["k_G"] = 1.66 * df["G"] ** -0.33
    y = np.log(df["d_equiv_um"])
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["tau_h"]),
        np.log(df["k_G"]),
        1.0 / df["T_K"],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, beta_tau, beta_kG, beta_T = beta
    y_pred = X @ beta
    d_pred = np.exp(y_pred)
    metrics = compute_metrics(df["d_equiv_um"], d_pred)
    model = {
        "intercept": intercept,
        "beta_tau": beta_tau,
        "beta_kG": beta_kG,
        "beta_T": beta_T,
        "metric": metrics,
    }
    if include_predictions:
        value = np.log(df["d_equiv_um"]) - intercept - beta_tau * np.log(df["tau_h"]) - beta_kG * np.log(df["k_G"])
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
    return 1.0 / denom


def estimate_temperature_kG(model, D, tau, G):
    if model["beta_T"] == 0:
        return None
    k_G = 1.66 * G ** -0.33
    value = math.log(D) - model["intercept"] - model["beta_tau"] * math.log(tau) - model["beta_kG"] * math.log(k_G)
    denom = value / model["beta_T"]
    if denom <= 0:
        return None
    return 1.0 / denom


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
        """
        Анализируем рост σ-фазы в стали 12Х18Н12Т, подбираем параметрические модели и
        получаем оценку температуры эксплуатации по твоим точкам. Загружай свежие CSV/XLSX
        в виджете слева — никаких встроенных данных, только то, что ты проверяешь прямо сейчас.
        """
    )

    with st.sidebar.expander("Загрузка данных"):
        uploaded = st.file_uploader("Загрузи CSV или Excel с измерениями", type=["csv", "xlsx", "xls"])
        st.caption("Обязательно: G, T_C, tau_h, dэкв_мкм; c_sigma_pct опционально")

    if uploaded is None:
        st.info("Загрузи файл с новыми экспериментальными точками — после загрузки появятся модели и графики.")
        return

    try:
        df = load_data(uploaded)
    except ValueError as exc:
        st.error(str(exc))
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
        st.markdown("### Модель с $k_G$")
        st.markdown(r"Исходный коэффициент зерна: $k_G = 1.66 \cdot G^{-0.33}$")
        st.latex(r"\ln D = a + b \ln \tau + c \ln k_G + \beta_T / T")
        st.markdown(
            fr"""
            - $b = {kG_model['beta_tau']:.3f}$
            - $c = {kG_model['beta_kG']:.3f}$
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
        if T_growth is not None:
            st.write(f"Ростовая модель: $T = {T_growth:.1f}\,K$ ({T_growth - 273.15:.1f} °C)")
        else:
            st.write("Ростовая модель не дала положительного решения")
        if T_kG is not None:
            st.write(f"Модель с $k_G$: $T = {T_kG:.1f}\,K$ ({T_kG - 273.15:.1f} °C)")
        else:
            st.write("k_G-модель не дала положительного решения")

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

    st.markdown("---")
    st.info(
        "– `m` лучше выбирать по графику RMSE (в боковой панели).\n"
        "– Загружай новые точки через uploader — приложение всегда работает с тем файлом, который ты сейчас проверяешь."
    )


if __name__ == "__main__":
    main()
