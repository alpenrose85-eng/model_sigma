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
MAX_PARTICLE_FACTOR = 4
SIGMA_F_MAX = 0.18  # максимальная доля σ-фазы (18%)
GRAIN_SIZES_MM = {
    -7: 4.0,
    -6: 2.828,
    -5: 2.0,
    -4: 1.414,
    -3: 1.0,
    -2: 0.707,
    -1: 0.5,
    0: 0.354,
    1: 0.25,
    2: 0.177,
    3: 0.125,
    4: 0.0884,
    5: 0.0625,
    6: 0.0442,
    7: 0.0312,
    8: 0.0221,
    9: 0.0156,
    10: 0.011,
}
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
        return {"rmse": float("nan"), "r2": float("nan")}
    rmse = math.sqrt(mean_squared_error(T_true[mask], T_pred[mask]))
    r2 = r2_score(T_true[mask], T_pred[mask])
    return {"rmse": rmse, "r2": r2}
def grain_diameter_um(G):
    G_int = int(round(float(G)))
    size_mm = GRAIN_SIZES_MM.get(G_int)
    return size_mm * 1000 if size_mm else 0.1
def grain_diameter_um_array(G_arr):
    G_np = np.array(G_arr, dtype=float)
    return np.vectorize(grain_diameter_um)(G_np)
def saturation_factor(D_kin, D_max):
    width = np.maximum(0.1, 0.05 * D_max)
    return 1.0 / (1.0 + np.exp((D_kin - D_max) / width))
def sigma_activity(T_K):
    T_C = T_K - 273.15
    lower = 1.0 / (1.0 + np.exp(-(T_C - SIGMA_TEMP_MIN) / SIGMA_TEMP_WIDTH))
    upper = 1.0 / (1.0 + np.exp((T_C - SIGMA_TEMP_MAX) / SIGMA_TEMP_WIDTH))
    return lower * upper
def compute_predicted_diameter(T_K, tau, G, model, m):
    k0 = model["k0"]
    Q_J = model["Q_J"]
    beta_G = model["beta_G"]
    gamma_T = sigma_activity(T_K)
    exponent = -Q_J / (R * T_K)
    D_kin = k0 * gamma_T * tau * np.exp(beta_G * G) * np.exp(exponent)
    return D_kin ** (1.0 / m)
def apply_saturation(D_kin, G):
    D_max = MAX_PARTICLE_FACTOR * grain_diameter_um(G)
    return D_kin * saturation_factor(D_kin, D_max)
def solve_temperature_for_growth(model, m, D, tau, G):
    def f(T):
        D_kin = compute_predicted_diameter(T, tau, G, model, m)
        return apply_saturation(D_kin, G) - D
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
def clamp_temperature(T_arr, return_bounds=False):
    min_K = SIGMA_TEMP_MIN + 273.15
    max_K = SIGMA_TEMP_MAX + 273.15
    if not return_bounds:
        return np.where((T_arr >= min_K) & (T_arr <= max_K), T_arr, np.nan)
    flags = np.where(T_arr < min_K, -1, np.where(T_arr > max_K, 1, 0))
    T_out = np.where((T_arr >= min_K) & (T_arr <= max_K), T_arr, np.nan)
    return T_out, flags
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
    df["grain_d_um"] = df["G"].apply(grain_diameter_um)
    y = np.log(df["d_equiv_um"])
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["tau_h"]),
        np.log(df["grain_d_um"].replace(0, 0.1)),
        1.0 / df["T_K"],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, beta_tau, beta_d, beta_T = beta
    y_pred = X @ beta
    d_pred = np.exp(y_pred)
    metrics = compute_metrics(df["d_equiv_um"], d_pred)
    model = {
        "intercept": intercept,
        "beta_tau": beta_tau,
        "beta_d": beta_d,
        "beta_T": beta_T,
        "metric": metrics,
    }
    if include_predictions:
        value = np.log(df["d_equiv_um"]) - intercept - beta_tau * np.log(df["tau_h"]) - beta_d * np.log(df["grain_d_um"].replace(0, 0.1))
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
    T_val = 1.0 / denom
    T_arr, flags = clamp_temperature(np.array([T_val]), return_bounds=True)
    if not np.isfinite(T_arr[0]):
        return "below" if flags[0] < 0 else "above"
    return float(T_arr[0])
def estimate_temperature_kG(model, D, tau, G):
    if model["beta_T"] == 0:
        return None
    dG_um = grain_diameter_um(G)
    value = math.log(D) - model["intercept"] - model["beta_tau"] * math.log(tau) - model["beta_d"] * math.log(dG_um)
    denom = value / model["beta_T"]
    if denom <= 0:
        return None
    T_val = 1.0 / denom
    T_arr, flags = clamp_temperature(np.array([T_val]), return_bounds=True)
    if not np.isfinite(T_arr[0]):
        return "below" if flags[0] < 0 else "above"
    return float(T_arr[0])
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
def predict_sigma_fraction(model, tau, G, T_K, D=None):
    y_pred = (
        model["intercept"]
        + model["beta_tau"] * np.log(tau)
        + model["beta_G"] * G
        + (model["beta_d"] * np.log(D) if D is not None else 0.0)
        + model["beta_T"] * (1.0 / T_K)
    )
    f_norm = 1.0 - np.exp(-np.exp(y_pred))
    return SIGMA_F_MAX * f_norm
def fit_sigma_fraction_model(df, include_d=False):
    df = df.copy()
    df = df[df["c_sigma_pct"].notna()].copy()
    df = df[(df["c_sigma_pct"] > 0) & (df["c_sigma_pct"] < 100)]
    if df.empty:
        return None
    f = df["c_sigma_pct"].values / 100.0
    f = np.minimum(f, SIGMA_F_MAX)
    f_norm = f / SIGMA_F_MAX
    y = np.log(-np.log(1.0 - f_norm))
    X_cols = [np.ones(len(df)), np.log(df["tau_h"]), df["G"], 1.0 / df["T_K"]]
    if include_d:
        X_cols.insert(3, np.log(df["d_equiv_um"]))
    X = np.column_stack(X_cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if include_d:
        intercept, beta_tau, beta_G, beta_d, beta_T = beta
    else:
        intercept, beta_tau, beta_G, beta_T = beta
        beta_d = 0.0
    model = {
        "intercept": intercept,
        "beta_tau": beta_tau,
        "beta_G": beta_G,
        "beta_d": beta_d,
        "beta_T": beta_T,
        "Q_kJ_per_mol": (-beta_T * R) / 1000.0,
        "df_index": df.index.to_numpy(),
    }
    # предсказание температуры на обучающей выборке
    T_pred = []
    for f_val, tau_val, G_val, d_val in zip(
        f, df["tau_h"], df["G"], df["d_equiv_um"]
    ):
        T_est = estimate_temperature_sigma(model, f_val, tau_val, G_val, d_val if include_d else None)
        if T_est in ("below", "above"):
            T_pred.append(np.nan)
        else:
            T_pred.append(np.nan if T_est is None else T_est)
    T_pred = np.array(T_pred, dtype=float)
    # предсказание f на обучающей выборке
    if include_d:
        f_pred = predict_sigma_fraction(model, df["tau_h"].values, df["G"].values, df["T_K"].values, df["d_equiv_um"].values)
    else:
        f_pred = predict_sigma_fraction(model, df["tau_h"].values, df["G"].values, df["T_K"].values, None)
    rmse_f = math.sqrt(mean_squared_error(f, f_pred))
    rmse_f_pct = rmse_f * 100.0
    r2_f = r2_score(f, f_pred)
    model["T_pred_K"] = T_pred
    model["f_pred"] = f_pred
    model["f_true"] = f
    model["metrics"] = compute_temp_metrics(df["T_K"].values, T_pred)
    model["metrics_f"] = {"rmse_frac": rmse_f, "rmse_pct": rmse_f_pct, "r2": r2_f}
    return model
def estimate_temperature_sigma(model, f_sigma, tau, G, D=None):
    if model is None:
        return None
    if f_sigma <= 0:
        return None
    f_sigma = min(f_sigma, SIGMA_F_MAX)
    f_norm = f_sigma / SIGMA_F_MAX
    if f_norm >= 1:
        return None
    y = math.log(-math.log(1.0 - f_norm))
    value = (
        model["intercept"]
        + model["beta_tau"] * math.log(tau)
        + model["beta_G"] * G
        + (model["beta_d"] * math.log(D) if D is not None and D > 0 else 0.0)
    )
    denom = (y - value) / model["beta_T"]
    if denom <= 0:
        return None
    T_val = 1.0 / denom
    T_arr, flags = clamp_temperature(np.array([T_val]), return_bounds=True)
    if not np.isfinite(T_arr[0]):
        return "below" if flags[0] < 0 else "above"
    return float(T_arr[0])
def render_analysis(df, selected_m):
    st.subheader("Параметры моделей")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Рост Dэкв (Аррениус)")
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
        st.markdown(
            """
        **Обозначения:**
        - **D** — эквивалентный диаметр σ‑фазы, мкм
        - **m** — показатель степени роста (подбирается по данным)
        - **k₀** — коэффициент, определяемый экспериментально
        - **Q** — энергия активации роста, кДж/моль
        - **R** — универсальная газовая постоянная = 8.314 Дж/(моль·К)
        - **T** — температура, К
        - **τ** — наработка, ч
        - **G** — номер зерна
        - **β_G** — коэффициент влияния зерна
            """
        )
    with col2:
        st.markdown("### Модель с $k_G$(предсказанный размер зерна)")
        st.markdown(r"Модель: $\ln D = a + b \ln \tau + c \ln d_G + \beta_T / T$, где $d_G$ — средний диаметр зерна")
        st.markdown(
            fr"""
            - $c = {kG_model['beta_d']:.3f}$ → $k_G \propto d_G^{{{kG_model['beta_d']:.3f}}}$
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
    # График модели D(P) с доверительным интервалом ±15%
    st.subheader("Рост Dэкв (Аррениус): D(P) с интервалом ±15%")
    P = df["T_K"] * (np.log10(df["tau_h"]) - 2 * np.log10(df["T_K"]) + 26.3)
    P_vals = P.values
    D_model = np.array(growth_model["D_pred"])
    order = np.argsort(P_vals)
    P_sorted = P_vals[order]
    D_sorted = D_model[order]
    D_low = D_sorted * 0.85
    D_high = D_sorted * 1.15
    fig_dp, ax_dp = plt.subplots(figsize=(8, 4))
    ax_dp.plot(P_sorted, D_sorted, color="tab:blue", label="Модель")
    ax_dp.fill_between(P_sorted, D_low, D_high, color="tab:blue", alpha=0.2, label="±15%")
    ax_dp.scatter(P_vals, df["d_equiv_um"], color="black", s=20, alpha=0.7, label="Эксперимент")
    ax_dp.set_xlabel("P = T*(log10 τ − 2 log10 T + 26.3)")
    ax_dp.set_ylabel("Dэкв, μm")
    ax_dp.set_title("Модель и экспериментальные точки")
    ax_dp.legend()
    st.pyplot(fig_dp)
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
    # Сводная таблица по 4 моделям температуры
    st.subheader("Сводная таблица качества (по диаметру D)")
    def subset_metrics_D(D_true, D_pred, mask):
        if not mask.any():
            return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
        rmse = math.sqrt(mean_squared_error(D_true[mask], D_pred[mask]))
        r2 = r2_score(D_true[mask], D_pred[mask])
        mae = np.mean(np.abs(D_true[mask] - D_pred[mask]))
        return {"rmse": rmse, "r2": r2, "mae": mae}
    T_true_C = df["T_C"].values
    focus_mask_temp = (T_true_C >= 580) & (T_true_C <= 650)
    models = [
        ("Рост Dэкв (Аррениус)", growth_model.get("D_pred")),
        ("Рост k_G (зерно)", kG_model.get("D_pred")),
    ]
    rows = []
    for name, preds in models:
        if preds is None:
            continue
        preds = np.array(preds)
        base = subset_metrics_D(df["d_equiv_um"].values, preds, np.isfinite(preds))
        focus_mask = np.isfinite(preds) & focus_mask_temp
        n_focus = int(focus_mask.sum())
        focus = subset_metrics_D(df["d_equiv_um"].values, preds, focus_mask) if n_focus >= 2 else {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
        render_analysis(df, selected_m)
        G_sel = st.selectbox("Номер зерна", sorted(df["G"].unique()))
    tab1, tab2, tab3 = st.tabs(["Анализ", "По зерну", "Калькулятор"])
    with tab1:
        render_analysis(df, selected_m)
    with tab2:
        G_sel = st.selectbox("Номер зерна", sorted(df["G"].unique()))
        df_g = df[df["G"] == G_sel].copy()
        if df_g.empty:
            st.info("Нет точек для выбранного номера зерна")
        else:
            render_analysis(df_g, selected_m)
    with tab3:
        st.markdown("Введите данные и получите температуру по 4 моделям")
        col1, col2 = st.columns(2)
        with col1:
            d_input_calc = st.number_input("D, мкм", min_value=0.1, value=1.5, format="%.3f", key="calc_d")
            c_sigma_calc = st.number_input("σ‑фаза, % (для JMAK)", min_value=0.1, max_value=SIGMA_F_MAX*100, value=5.0, format="%.2f", key="calc_sigma")
        with col2:
            tau_calc = st.number_input("τ, часы", min_value=1.0, value=5000.0, format="%.1f", key="calc_tau")
            G_calc = st.number_input("G (номер зерна)", min_value=1.0, value=8.0, format="%.1f", key="calc_g")
        if st.button("Рассчитать", key="calc_btn"):
            T1 = estimate_temperature_growth(growth_model, d_input_calc, tau_calc, G_calc, selected_m)
            T2 = estimate_temperature_kG(kG_model, d_input_calc, tau_calc, G_calc)
            T3 = estimate_temperature_sigma(sigma_model_basic, c_sigma_calc/100.0, tau_calc, G_calc, None)
            T4 = estimate_temperature_sigma(sigma_model_with_d, c_sigma_calc/100.0, tau_calc, G_calc, d_input_calc)
            st.markdown("**Результаты (K / °C):**")
            def fmt(T):
                if T == "below":
                    return "< 560°C"
                if T == "above":
                    return "> 900°C"
                return "—" if T is None else f"{T:.1f} K ({T-273.15:.1f} °C)"
            st.write(f"Рост Dэкв (Аррениус): {fmt(T1)}")
            st.write(f"Рост k_G (зерно): {fmt(T2)}")
            st.write(f"JMAK без D: {fmt(T3)}")
            st.write(f"JMAK с D: {fmt(T4)}")
if __name__ == "__main__":
    main()
