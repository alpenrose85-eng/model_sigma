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
    Q_J = model["Q_J"]
    beta_G = model["beta_G"]
    gamma_T = sigma_activity(T_K)
    exponent = -Q_J / (R * T_K)
    D_kin = k0 * gamma_T * tau * np.exp(beta_G * G) * np.exp(exponent)
    return D_kin ** (1.0 / m)


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
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
    rmse = math.sqrt(mean_squared_error(T_true[mask], T_pred[mask]))
    r2 = r2_score(T_true[mask], T_pred[mask])
    mae = np.mean(np.abs(T_true[mask] - T_pred[mask]))
    return {"rmse": rmse, "r2": r2, "mae": mae}


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


def fit_growth_model(df, m, include_predictions=False, include_G=True):
    df = df.copy()
    y = m * np.log(df["d_equiv_um"]) - np.log(df["tau_h"])
    if include_G:
        X = np.column_stack([
            np.ones(len(df)),
            1.0 / df["T_K"],
            df["G"],
        ])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, beta_T, beta_G = beta
    else:
        X = np.column_stack([
            np.ones(len(df)),
            1.0 / df["T_K"],
        ])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, beta_T = beta
        beta_G = 0.0
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


def fit_kG_model(df, include_predictions=False, include_G=True):
    df = df.copy()
    df["grain_d_um"] = df["G"].apply(grain_diameter_um)
    y = np.log(df["d_equiv_um"])
    if include_G:
        X = np.column_stack([
            np.ones(len(df)),
            np.log(df["tau_h"]),
            np.log(df["grain_d_um"].replace(0, 0.1)),
            1.0 / df["T_K"],
        ])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, beta_tau, beta_d, beta_T = beta
    else:
        X = np.column_stack([
            np.ones(len(df)),
            np.log(df["tau_h"]),
            1.0 / df["T_K"],
        ])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept, beta_tau, beta_T = beta
        beta_d = 0.0
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


def fit_inverse_temp_model(df, include_G=True):
    df = df.copy()
    c_sigma = df["c_sigma_pct"].replace(0, 0.1)
    if include_G:
        features = np.column_stack([
            np.log(df["d_equiv_um"]),
            np.log(df["tau_h"]),
            df["G"],
            np.log(c_sigma),
        ])
    else:
        features = np.column_stack([
            np.log(df["d_equiv_um"]),
            np.log(df["tau_h"]),
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


def fit_boosted_temp_model(df, include_G=True):
    df = df.copy()
    df["c_sigma_pct"] = df["c_sigma_pct"].replace(0, 0.1)
    if include_G:
        features = df[["d_equiv_um", "tau_h", "G", "c_sigma_pct"]]
    else:
        features = df[["d_equiv_um", "tau_h", "c_sigma_pct"]]
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


def fit_sigma_fraction_model(df, include_d=False, include_G=True):
    df = df.copy()
    df = df[df["c_sigma_pct"].notna()].copy()
    df = df[(df["c_sigma_pct"] > 0) & (df["c_sigma_pct"] < 100)]
    if df.empty:
        return None
    f = df["c_sigma_pct"].values / 100.0
    f = np.minimum(f, SIGMA_F_MAX)
    f_norm = f / SIGMA_F_MAX
    y = np.log(-np.log(1.0 - f_norm))
    X_cols = [np.ones(len(df)), np.log(df["tau_h"]), 1.0 / df["T_K"]]
    if include_G:
        X_cols.insert(2, df["G"])
    if include_d:
        insert_idx = 3 if include_G else 2
        X_cols.insert(insert_idx, np.log(df["d_equiv_um"]))
    X = np.column_stack(X_cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if include_G and include_d:
        intercept, beta_tau, beta_G, beta_d, beta_T = beta
    elif include_G and not include_d:
        intercept, beta_tau, beta_G, beta_T = beta
        beta_d = 0.0
    elif (not include_G) and include_d:
        intercept, beta_tau, beta_d, beta_T = beta
        beta_G = 0.0
    else:
        intercept, beta_tau, beta_T = beta
        beta_G = 0.0
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


def deviation_color(v):
    if pd.isna(v):
        return ""
    if v <= 15:
        return "background-color: #dff5e1"
    if v <= 25:
        return "background-color: #fff3cd"
    return "background-color: #f8d7da"


def filter_table(df, dev_col, dev_values, key_prefix):
    df_edit = df[["G", "T_C", "tau_h", "d_equiv_um", "c_sigma_pct"]].copy()
    df_edit["d_equiv_um"] = df_edit["d_equiv_um"].round(3)
    df_edit["c_sigma_pct"] = df_edit["c_sigma_pct"].round(2)
    df_edit.insert(0, "exclude", False)
    df_edit["row_id"] = df_edit.index.astype(int)
    df_edit[dev_col] = dev_values
    df_edit[dev_col] = df_edit[dev_col].replace([np.inf, -np.inf], np.nan).round(0)

    styled = df_edit.style.applymap(deviation_color, subset=[dev_col]).format(
        {
            "d_equiv_um": "{:.3f}",
            "c_sigma_pct": "{:.2f}",
            dev_col: "{:.0f}",
        }
    )
    st.markdown("**Цветная оценка отклонений (|Δ|, %):**")
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("**Таблица выбора точек для исключения:**")
    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key=f"exclude_editor_{key_prefix}",
    )
    exclude_ids = edited.loc[edited["exclude"] == True, "row_id"].tolist()
    if exclude_ids:
        st.info(f"Исключено точек: {len(exclude_ids)}")
    return df.drop(index=exclude_ids)


def render_growth_tab(df, selected_m, key_prefix="growth"):
    st.subheader("Рост Dэкв (Аррениус)")
    st.latex(r"D^m = k_0 \cdot e^{-Q/(RT)} \cdot \tau \cdot e^{\beta_G G}")
    st.markdown(
        """
**Обозначения:**
- **D** — эквивалентный диаметр σ‑фазы, мкм
- **m** — показатель степени роста
- **k₀** — коэффициент
- **Q** — энергия активации, кДж/моль
- **R** — газовая постоянная
- **T** — температура, К
- **τ** — наработка, ч
- **G** — номер зерна
- **β_G** — коэффициент влияния зерна
"""
    )

    include_G = df["G"].nunique() > 1
    initial = fit_growth_model(df, selected_m, include_predictions=True, include_G=include_G)
    dev = np.abs(df["d_equiv_um"].values - initial["D_pred"]) / df["d_equiv_um"].values * 100
    df_f = filter_table(df, "|ΔD|, %", dev, key_prefix)
    include_G = df_f["G"].nunique() > 1
    model = fit_growth_model(df_f, selected_m, include_predictions=True, include_G=include_G)

    st.markdown(
        fr"""
- $k_0 = e^{{{model['intercept']:.2f}}} \approx {model['k0']:.0f}$
- $Q = {-model['beta_T'] * R / 1000:.1f}\,\mathrm{{кДж/моль}}$
- $\beta_G = {model['beta_G']:.3f}$
- $\mathrm{{RMSE}}(D) = {model['metric']['rmse']:.3f}\,\mu\mathrm{{m}}$
- $\mathrm{{RMSE}}(\ln D) = {model['metric']['rmse_log']:.3f}$
- $R^2 = {model['metric']['r2']:.3f}$
- $\mathrm{{RMSE}}(T) = {model['temp_rmse_K']:.1f}\,\mathrm{{°C}}$
""",
        unsafe_allow_html=True,
    )

    st.subheader("Графики качества")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].scatter(df_f["d_equiv_um"], model["D_pred"], c="tab:blue")
    axs[0].plot(df_f["d_equiv_um"], df_f["d_equiv_um"], linestyle="--", color="gray")
    axs[0].set_xlabel("Наблюдаемый D, μm")
    axs[0].set_ylabel("Предсказанный D")
    axs[0].set_title("Факт vs прогноз")
    axs[1].hist(df_f["d_equiv_um"] - model["D_pred"], bins=15, color="tab:blue")
    axs[1].set_xlabel("ΔD, μm")
    axs[1].set_title("Остатки")
    st.pyplot(fig)

    st.subheader("D(P) с интервалом ±15%")
    P = df_f["T_K"] * (np.log10(df_f["tau_h"]) - 2 * np.log10(df_f["T_K"]) + 26.3)
    P_vals = P.values
    D_model = np.array(model["D_pred"])
    order = np.argsort(P_vals)
    P_sorted = P_vals[order]
    D_sorted = D_model[order]
    D_low = D_sorted * 0.85
    D_high = D_sorted * 1.15
    fig_dp, ax_dp = plt.subplots(figsize=(8, 4))
    ax_dp.plot(P_sorted, D_sorted, color="tab:blue", label="Модель")
    ax_dp.fill_between(P_sorted, D_low, D_high, color="tab:blue", alpha=0.2, label="±15%")
    ax_dp.scatter(P_vals, df_f["d_equiv_um"], color="black", s=20, alpha=0.7, label="Эксперимент")
    ax_dp.set_xlabel("P = T*(log10 τ − 2 log10 T + 26.3)")
    ax_dp.set_ylabel("Dэкв, μm")
    ax_dp.set_title("Модель и экспериментальные точки")
    ax_dp.legend()
    st.pyplot(fig_dp)

    st.subheader("Температура: прогноз vs факт")
    fig_temp, ax_temp = plt.subplots(figsize=(6, 5))
    true_T = df_f["T_C"]
    preds = model["T_pred_K"] - 273.15
    mask = np.isfinite(preds)
    ax_temp.scatter(true_T[mask], preds[mask], color="tab:blue", alpha=0.7)
    ax_temp.plot(true_T, true_T, linestyle="--", color="gray")
    ax_temp.set_xlabel("Наблюдаемая T, °C")
    ax_temp.set_ylabel("Предсказанная T, °C")
    ax_temp.set_title("Ростовая модель")
    st.pyplot(fig_temp)

    st.subheader("Качество модели")
    quality = {
        "RMSE D, μm": model["metric"]["rmse"],
        "R² D": model["metric"]["r2"],
        "RMSE T, °C": model["temp_rmse_K"],
    }
    st.dataframe(pd.DataFrame([quality]))


def render_kG_tab(df, key_prefix="kG"):
    st.subheader("Рост с k_G (зерно)")
    st.latex(r"\ln D = a + b\ln\tau + c\ln d_G + \beta_T/T")
    st.markdown(
        """
**Обозначения:**
- **D** — эквивалентный диаметр, мкм
- **τ** — наработка
- **d_G** — средний диаметр зерна
- **T** — температура, К
"""
    )

    include_G = df["G"].nunique() > 1
    initial = fit_kG_model(df, include_predictions=True, include_G=include_G)
    dev = np.abs(df["d_equiv_um"].values - initial["D_pred"]) / df["d_equiv_um"].values * 100
    df_f = filter_table(df, "|ΔD|, %", dev, key_prefix)
    include_G = df_f["G"].nunique() > 1
    model = fit_kG_model(df_f, include_predictions=True, include_G=include_G)

    st.markdown(
        fr"""
- $c = {model['beta_d']:.3f}$ → $k_G \propto d_G^{{{model['beta_d']:.3f}}}$
- $b = {model['beta_tau']:.3f}$
- $\beta_T = {model['beta_T']:.1f}$
- $\mathrm{{RMSE}}(D) = {model['metric']['rmse']:.3f}\,\mu\mathrm{{m}}$
- $\mathrm{{RMSE}}(\ln D) = {model['metric']['rmse_log']:.3f}$
- $R^2 = {model['metric']['r2']:.3f}$
- $\mathrm{{RMSE}}(T) = {model['temp_rmse_K']:.1f}\,\mathrm{{°C}}$
""",
        unsafe_allow_html=True,
    )

    st.subheader("Графики качества")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axs[0].scatter(df_f["d_equiv_um"], model["D_pred"], c="tab:orange")
    axs[0].plot(df_f["d_equiv_um"], df_f["d_equiv_um"], linestyle="--", color="gray")
    axs[0].set_xlabel("Наблюдаемый D, μm")
    axs[0].set_ylabel("Предсказанный D")
    axs[0].set_title("Факт vs прогноз")
    axs[1].hist(df_f["d_equiv_um"] - model["D_pred"], bins=15, color="tab:orange")
    axs[1].set_xlabel("ΔD, μm")
    axs[1].set_title("Остатки")
    st.pyplot(fig)

    st.subheader("Температура: прогноз vs факт")
    fig_temp, ax_temp = plt.subplots(figsize=(6, 5))
    true_T = df_f["T_C"]
    preds = model["T_pred_K"] - 273.15
    mask = np.isfinite(preds)
    ax_temp.scatter(true_T[mask], preds[mask], color="tab:orange", alpha=0.7)
    ax_temp.plot(true_T, true_T, linestyle="--", color="gray")
    ax_temp.set_xlabel("Наблюдаемая T, °C")
    ax_temp.set_ylabel("Предсказанная T, °C")
    ax_temp.set_title("k_G модель")
    st.pyplot(fig_temp)

    st.subheader("Качество модели")
    quality = {
        "RMSE D, μm": model["metric"]["rmse"],
        "R² D": model["metric"]["r2"],
        "RMSE T, °C": model["temp_rmse_K"],
    }
    st.dataframe(pd.DataFrame([quality]))


def render_inverse_tab(df, key_prefix="inverse"):
    st.subheader("Регрессия по 1/T")
    st.latex(r"\frac{1}{T} = a + b\ln D + c\ln\tau + dG + e\ln c_{\sigma}")
    st.markdown("Непараметрическая линейная модель по обратной температуре.")

    include_G = df["G"].nunique() > 1
    initial = fit_inverse_temp_model(df, include_G=include_G)
    T_true = df["T_C"].values
    preds_init = initial["T_pred_K"] - 273.15
    dev = np.abs(T_true - preds_init) / T_true * 100
    df_f = filter_table(df, "|ΔT|, %", dev, key_prefix)
    include_G = df_f["G"].nunique() > 1
    model = fit_inverse_temp_model(df_f, include_G=include_G)

    st.markdown(f"RMSE T: {model['metrics']['rmse']:.2f} °C, R²: {model['metrics']['r2']:.3f}")

    st.subheader("Графики качества")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    preds = model["T_pred_K"] - 273.15
    mask = np.isfinite(preds)
    axs[0].scatter(df_f["T_C"][mask], preds[mask], color="tab:green", alpha=0.7)
    axs[0].plot(df_f["T_C"], df_f["T_C"], linestyle="--", color="gray")
    axs[0].set_xlabel("Наблюдаемая T, °C")
    axs[0].set_ylabel("Предсказанная T, °C")
    axs[0].set_title("Факт vs прогноз")
    axs[1].hist(df_f["T_C"][mask] - preds[mask], bins=15, color="tab:green")
    axs[1].set_xlabel("ΔT, °C")
    axs[1].set_title("Остатки")
    st.pyplot(fig)


def render_boosted_tab(df, key_prefix="boost"):
    st.subheader("Градиентный бустинг")
    st.markdown("Функция: $T = f(D, \tau, G, c_{\sigma})$")
    st.markdown(
        """
Ансамблевый метод на деревьях решений. Дает высокую точность, но без аналитической формулы.
"""
    )

    include_G = df["G"].nunique() > 1
    initial = fit_boosted_temp_model(df, include_G=include_G)
    T_true = df["T_C"].values
    preds_init = initial["T_pred_K"] - 273.15
    dev = np.abs(T_true - preds_init) / T_true * 100
    df_f = filter_table(df, "|ΔT|, %", dev, key_prefix)
    include_G = df_f["G"].nunique() > 1
    model = fit_boosted_temp_model(df_f, include_G=include_G)

    st.markdown(f"RMSE T: {model['metrics']['rmse']:.2f} °C, R²: {model['metrics']['r2']:.3f}")

    try:
        importances = model["model"].feature_importances_
        feat_names = ["D", "τ", "G", "cσ"] if df_f["G"].nunique() > 1 else ["D", "τ", "cσ"]
        fig_imp, ax_imp = plt.subplots(figsize=(4, 3))
        ax_imp.bar(feat_names, importances)
        ax_imp.set_title("Важность признаков")
        st.pyplot(fig_imp)
    except Exception:
        pass

    st.subheader("Графики качества")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    preds = model["T_pred_K"] - 273.15
    mask = np.isfinite(preds)
    axs[0].scatter(df_f["T_C"][mask], preds[mask], color="tab:red", alpha=0.7)
    axs[0].plot(df_f["T_C"], df_f["T_C"], linestyle="--", color="gray")
    axs[0].set_xlabel("Наблюдаемая T, °C")
    axs[0].set_ylabel("Предсказанная T, °C")
    axs[0].set_title("Факт vs прогноз")
    axs[1].hist(df_f["T_C"][mask] - preds[mask], bins=15, color="tab:red")
    axs[1].set_xlabel("ΔT, °C")
    axs[1].set_title("Остатки")
    st.pyplot(fig)


def render_sigma_tab(df, include_d, key_prefix="sigma"):
    label = "JMAK с D" if include_d else "JMAK без D"
    st.subheader(f"{label}")
    st.latex(r"f_{\sigma}^{max}=0.18,\quad f_{\sigma}=f_{\sigma}^{max}(1-\exp[-k(T)\,\tau^{n}])")
    st.latex(r"\ln\left[-\ln\left(1-\frac{f_{\sigma}}{f_{\sigma}^{max}}\right)\right]= a + b\ln\tau + cG + d\ln D + \beta_T/T")
    st.markdown(
        """
JMAK описывает кинетику выделения фаз, ограниченную максимумом 18% σ.
Используется для оценки температуры по содержанию σ‑фазы.
"""
    )

    df_sigma = df[df["c_sigma_pct"].notna()].copy()
    if df_sigma.empty:
        st.info("Нет данных по содержанию σ‑фазы.")
        return

    include_G = df_sigma["G"].nunique() > 1
    initial = fit_sigma_fraction_model(df_sigma, include_d=include_d, include_G=include_G)
    if initial is None:
        st.info("Недостаточно данных для JMAK модели.")
        return

    f_pred = initial["f_pred"] * 100
    f_true = df_sigma.loc[initial["df_index"], "c_sigma_pct"].values
    dev = np.abs(f_true - f_pred) / np.where(f_pred == 0, np.nan, f_pred) * 100
    df_f = filter_table(df_sigma.loc[initial["df_index"]], "|Δσ|, %", dev, key_prefix)
    include_G = df_f["G"].nunique() > 1
    model = fit_sigma_fraction_model(df_f, include_d=include_d, include_G=include_G)
    if model is None:
        st.info("Недостаточно данных после фильтрации.")
        return

    st.markdown(
        fr"""
- $Q = {model['Q_kJ_per_mol']:.1f}\,\mathrm{{кДж/моль}}$
- $\mathrm{{RMSE}}(\%\sigma) = {model['metrics_f']['rmse_pct']:.2f}$
- $R^2 = {model['metrics_f']['r2']:.3f}$
""",
        unsafe_allow_html=True,
    )

    st.subheader("Графики качества")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    max_pct = SIGMA_F_MAX * 100
    axs[0].scatter(model["f_true"] * 100, model["f_pred"] * 100, color="tab:purple", alpha=0.7)
    axs[0].plot([0, max_pct], [0, max_pct], linestyle="--", color="gray")
    axs[0].set_xlim(0, max_pct)
    axs[0].set_ylim(0, max_pct)
    axs[0].set_xlabel("Факт, %")
    axs[0].set_ylabel("Прогноз, %")
    axs[0].set_title("%σ: факт vs прогноз")
    errs = (model["f_true"] - model["f_pred"]) * 100
    axs[1].hist(errs, bins=15, color="tab:purple")
    axs[1].set_xlabel("Δ%σ")
    axs[1].set_title("Остатки")
    st.pyplot(fig)

    st.subheader("Качество модели")
    sigma_df = pd.DataFrame([
        {
            "RMSE, %": model["metrics_f"]["rmse_pct"],
            "R²": model["metrics_f"]["r2"],
            "Макс. отклонение, %": np.nanmax(np.abs(errs)) if len(errs) else np.nan,
            "N": len(errs),
        }
    ])
    st.dataframe(sigma_df.style.format({"RMSE, %": "{:.2f}", "R²": "{:.3f}", "Макс. отклонение, %": "{:.2f}"}))


def render_summary_tab(df, selected_m):
    st.subheader("Сводка качества по всем моделям")
    include_G = df["G"].nunique() > 1
    growth_model = fit_growth_model(df, selected_m, include_predictions=True, include_G=include_G)
    kG_model = fit_kG_model(df, include_predictions=True, include_G=include_G)
    inverse_model = fit_inverse_temp_model(df, include_G=include_G)
    boosted_model = fit_boosted_temp_model(df, include_G=include_G)
    sigma_model_basic = fit_sigma_fraction_model(df, include_d=False, include_G=include_G)
    sigma_model_with_d = fit_sigma_fraction_model(df, include_d=True, include_G=include_G)

    st.markdown("### Качество предсказания диаметра D")
    rows_D = []
    for name, preds in [
        ("Рост Dэкв (Аррениус)", growth_model.get("D_pred")),
        ("Рост k_G (зерно)", kG_model.get("D_pred")),
    ]:
        if preds is None:
            continue
        base = compute_metrics(df["d_equiv_um"].values, np.array(preds))
        rows_D.append({
            "Модель": name,
            "R²": base["r2"],
            "RMSE, μm": base["rmse"],
            "RMSE lnD": base["rmse_log"],
        })
    if rows_D:
        st.dataframe(pd.DataFrame(rows_D).style.format({"R²": "{:.3f}", "RMSE, μm": "{:.3f}", "RMSE lnD": "{:.3f}"}))
    else:
        st.info("Нет моделей диаметра для сводки.")

    st.markdown("### Качество предсказания температуры T")
    rows_T = []
    T_true_C = df["T_C"].values
    focus_mask = (T_true_C >= 580) & (T_true_C <= 650)
    for name, preds in [
        ("Рост Dэкв (Аррениус)", growth_model.get("T_pred_K")),
        ("Рост k_G (зерно)", kG_model.get("T_pred_K")),
        ("Регрессия 1/T", inverse_model.get("T_pred_K")),
        ("Градиентный бустинг", boosted_model.get("T_pred_K")),
    ]:
        if preds is None:
            continue
        preds_C = np.array(preds) - 273.15
        metrics = compute_temp_metrics(T_true_C, preds_C)
        focus_metrics = compute_temp_metrics(T_true_C[focus_mask], preds_C[focus_mask]) if focus_mask.sum() >= 2 else {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}
        rows_T.append({
            "Модель": name,
            "R²": metrics["r2"],
            "RMSE, °C": metrics["rmse"],
            "MAE, °C": metrics["mae"],
            "R² (580–650°C)": focus_metrics["r2"],
            "RMSE, °C (580–650°C)": focus_metrics["rmse"],
            "MAE, °C (580–650°C)": focus_metrics["mae"],
            "N (580–650°C)": int(focus_mask.sum()),
        })
    if rows_T:
        st.dataframe(pd.DataFrame(rows_T).style.format({
            "R²": "{:.3f}",
            "RMSE, °C": "{:.1f}",
            "MAE, °C": "{:.1f}",
            "R² (580–650°C)": "{:.3f}",
            "RMSE, °C (580–650°C)": "{:.1f}",
            "MAE, °C (580–650°C)": "{:.1f}",
        }))
    else:
        st.info("Нет температурных моделей для сводки.")

    st.markdown("### Качество моделей по содержанию σ‑фазы")
    sigma_rows = []
    for name_m, mdl in [("JMAK без D", sigma_model_basic), ("JMAK с D", sigma_model_with_d)]:
        if mdl is None:
            continue
        err_pct = np.abs(mdl["f_true"] - mdl["f_pred"]) * 100
        sigma_rows.append({
            "Модель": name_m,
            "RMSE, %": mdl["metrics_f"]["rmse_pct"],
            "R²": mdl["metrics_f"]["r2"],
            "Макс. отклонение, %": np.nanmax(err_pct),
            "N": len(err_pct),
        })
    if sigma_rows:
        st.dataframe(pd.DataFrame(sigma_rows).style.format({"RMSE, %": "{:.2f}", "R²": "{:.3f}", "Макс. отклонение, %": "{:.2f}"}))
    else:
        st.info("Нет данных для оценки σ‑фазы.")


def render_calculator(df, selected_m):
    include_G = df["G"].nunique() > 1
    growth_model = fit_growth_model(df, selected_m, include_predictions=True, include_G=include_G)
    kG_model = fit_kG_model(df, include_predictions=True, include_G=include_G)
    boosted_model = fit_boosted_temp_model(df, include_G=include_G)
    sigma_model_basic = fit_sigma_fraction_model(df, include_d=False, include_G=include_G)
    sigma_model_with_d = fit_sigma_fraction_model(df, include_d=True, include_G=include_G)

    st.markdown("Введите данные и получите температуру по моделям")
    col1, col2 = st.columns(2)
    with col1:
        d_input_calc = st.number_input("D, мкм", min_value=0.1, value=1.5, format="%.3f", key="calc_d")
        c_sigma_calc = st.number_input("σ‑фаза, % (для JMAK)", min_value=0.1, max_value=SIGMA_F_MAX * 100, value=5.0, format="%.2f", key="calc_sigma")
    with col2:
        tau_calc = st.number_input("τ, часы", min_value=1.0, value=5000.0, format="%.1f", key="calc_tau")
        G_calc = st.number_input("G (номер зерна)", min_value=1.0, value=8.0, format="%.1f", key="calc_g")
    if st.button("Рассчитать", key="calc_btn"):
        T1 = estimate_temperature_growth(growth_model, d_input_calc, tau_calc, G_calc, selected_m)
        T2 = estimate_temperature_kG(kG_model, d_input_calc, tau_calc, G_calc)
        T3 = estimate_temperature_sigma(sigma_model_basic, c_sigma_calc / 100.0, tau_calc, G_calc, None)
        T4 = estimate_temperature_sigma(sigma_model_with_d, c_sigma_calc / 100.0, tau_calc, G_calc, d_input_calc)
        try:
            feat = np.array([[d_input_calc, tau_calc, G_calc, c_sigma_calc]])
            T5 = boosted_model["model"].predict(feat)[0]
        except Exception:
            T5 = None

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
        st.write(f"Градиентный бустинг: {fmt(T5)}")


def render_model_tabs(df, selected_m, key_prefix="main"):
    tabs = st.tabs([
        "Сводка",
        "Рост Dэкв",
        "Рост k_G",
        "1/T",
        "Бустинг",
        "JMAK без D",
        "JMAK с D",
    ])
    with tabs[0]:
        render_summary_tab(df, selected_m)
    with tabs[1]:
        render_growth_tab(df, selected_m, key_prefix=f"{key_prefix}_growth")
    with tabs[2]:
        render_kG_tab(df, key_prefix=f"{key_prefix}_kG")
    with tabs[3]:
        render_inverse_tab(df, key_prefix=f"{key_prefix}_inverse")
    with tabs[4]:
        render_boosted_tab(df, key_prefix=f"{key_prefix}_boost")
    with tabs[5]:
        render_sigma_tab(df, include_d=False, key_prefix=f"{key_prefix}_sigma_basic")
    with tabs[6]:
        render_sigma_tab(df, include_d=True, key_prefix=f"{key_prefix}_sigma_d")


def main():
    st.title("Sigma-phase temperature model")
    st.markdown(
        f"""
        Анализируем рост σ-фазы в стали 12Х18Н12Т, подбираем параметрические модели и
        получаем оценку температуры эксплуатации по твоим точкам. Загружай свежие CSV/XLSX
        в виджете слева — никаких встроенных данных, только то, что ты проверяешь прямо сейчас.
        σ-фаза стабилизируется начиная примерно {SIGMA_TEMP_MIN}°C и растворяется ближе к
        {SIGMA_TEMP_MAX}°C, поэтому все вычисления выполняются только внутри этого диапазона.
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

    tab_main, tab_grain, tab_calc = st.tabs(["Модели", "По зерну", "Калькулятор"])
    with tab_main:
        render_model_tabs(df, selected_m, key_prefix="all")
    with tab_grain:
        G_sel = st.selectbox("Номер зерна", sorted(df["G"].unique()))
        df_g = df[df["G"] == G_sel].copy()
        if df_g.empty:
            st.info("Нет точек для выбранного номера зерна")
        else:
            render_model_tabs(df_g, selected_m, key_prefix=f"G_{G_sel}")
    with tab_calc:
        render_calculator(df, selected_m)


if __name__ == "__main__":
    main()
