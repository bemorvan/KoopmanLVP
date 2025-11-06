# koopman_lpv_flowline_v3.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# =============================================================
# Koopman LPV for Parallel+Merge Flowline (v3)
# - Lit les CSV (xA,xB,xC + nu/a/q/lambda) depuis <racine>/data2
# - Dataset à fenêtres glissantes (X, X_next, Theta)
# - Apprentissage K(θ) = sum_j psi_j(θ) K_j (EDMD LPV)
# - Tracés de comparaison vrai vs prédiction
# - États = niveaux (pièces) ⇒ clamp [0, x_max] (pas de normalisation de somme)
# =============================================================

PathLike = Union[str, Path]
DEFAULT_STATE_COLS = ["xA", "xB", "xC"]
DEFAULT_THETA_COLS = [
    "nu1", "nu2", "nu3",
    "a1", "a2", "a3",
    "q1", "q2", "q3",
    "lambda1_in", "lambda2_in",
]

# -----------------------------
# Trouver <racine>/data2 (jamais dans src/scr)
# -----------------------------
def _resolve_data2_root() -> Path:
    env = os.environ.get("FLOWLINE_DATA_DIR", "").strip()
    if env:
        d = Path(env).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        d = base / "data2"
        if d.is_dir():
            return d.resolve()
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = cwd
    base_guess = script_dir.parent if script_dir.name.lower() in {"src","scr"} else cwd
    d = (base_guess / "data2").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d

# -----------------------------
# Dataset structure
# -----------------------------
@dataclass
class LPVDatasetFlow:
    X: np.ndarray        # (n_traj, n_steps, n_states)
    X_next: np.ndarray   # (n_traj, n_steps, n_states)
    Theta: np.ndarray    # (n_traj, n_steps, m_features)
    x0: np.ndarray       # (n_traj, n_states)
    t: np.ndarray        # (n_steps+1,) timestamps starting at 0
    dt: float
    n_steps: int
    state_names: List[str]
    theta_names: List[str]
    x_max: Optional[np.ndarray] = None  # bornes sup d'états (pour clamp)

# -----------------------------
# Chargement CSV (exclut les "combinés")
# -----------------------------
def load_flow_csvs(paths: Optional[Iterable[PathLike]] = None,
                   data_dir: Optional[PathLike] = None) -> pd.DataFrame:
    if paths is None:
        d = Path(data_dir).expanduser().resolve() if data_dir else _resolve_data2_root()
        csvs = sorted(
            p for p in d.glob("flowline_*.csv")
            if not p.name.endswith(("all_scenarios.csv", "piecewise_bank.csv"))
        )
        if not csvs:
            raise FileNotFoundError(f"Aucun flowline_*.csv trouvé dans {d}")
        paths = csvs
    dfs: List[pd.DataFrame] = []
    for i, p in enumerate(paths):
        pth = Path(p).expanduser().resolve()
        df = pd.read_csv(pth)
        if "scenario" not in df.columns:
            df["scenario"] = pth.stem.replace("flowline_", "")
        df["trace_id"] = pth.stem          # <-- identifiant unique par fichier
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    if "time_s" in data.columns:
        data["time_s"] = pd.to_numeric(data["time_s"], errors="coerce")
    return data.sort_values(["scenario","time_s"]).reset_index(drop=True)

# -----------------------------
# Fenêtrage → dataset LPV
# -----------------------------
def build_lpv_dataset_from_df(df: pd.DataFrame,
                              state_cols: Sequence[str] = DEFAULT_STATE_COLS,
                              theta_cols: Sequence[str] = DEFAULT_THETA_COLS,
                              n_steps: int = 600,
                              stride: int = 300,
                              dt: Optional[float] = None,
                              by_scenario: bool = True,
                              drop_nan: bool = True,
                              x_max: Optional[Sequence[float]] = None) -> LPVDatasetFlow:
    req = set(state_cols) | set(theta_cols) | {"time_s","scenario"}
    if not req.issubset(df.columns):
        miss = sorted(req - set(df.columns))
        raise ValueError(f"Colonnes manquantes: {miss}")
    if drop_nan:
        df = df.dropna(subset=list(req))

    groups = [df]
    if "trace_id" in df.columns:
        groups = [g for _, g in df.groupby("trace_id")]
    elif by_scenario:
        groups = [g for _, g in df.groupby("scenario")]


    Xs: List[np.ndarray] = []; Xns: List[np.ndarray] = []
    Ths: List[np.ndarray] = []; x0s: List[np.ndarray] = []
    t_ref: Optional[np.ndarray] = None; used_dt: Optional[float] = None

    x_max_arr: Optional[np.ndarray] = np.asarray(x_max, dtype=float) if x_max is not None else None

    for g in groups:
        g = g.sort_values("time_s").reset_index(drop=True)
        dt_g = float(np.median(np.diff(g["time_s"])) if len(g) > 1 else 1.0) if dt is None else float(dt)
        S = g[list(state_cols)].to_numpy(float)
        Th = g[list(theta_cols)].to_numpy(float)
        T = len(g); wnd = n_steps + 1
        for start in range(0, T - wnd + 1, stride):
            end = start + wnd
            X_full = S[start:end]; Th_full = Th[start:end-1]
            Xs.append(X_full[:-1]); Xns.append(X_full[1:]); Ths.append(Th_full); x0s.append(X_full[0])
        if t_ref is None:
            t_ref = np.arange(n_steps+1, dtype=float) * dt_g
            used_dt = dt_g
        if x_max_arr is None:
            mx = S.max(axis=0); x_max_arr = np.maximum(mx, 1.0) * 1.1

    if not Xs:
        raise RuntimeError("Aucune fenêtre créée (vérifie n_steps/stride).")

    X = np.stack(Xs, axis=0); Xn = np.stack(Xns, axis=0)
    Theta = np.stack(Ths, axis=0); x0 = np.stack(x0s, axis=0)
    return LPVDatasetFlow(X=X, X_next=Xn, Theta=Theta, x0=x0,
                          t=t_ref if t_ref is not None else np.arange(n_steps+1)*1.0,
                          dt=used_dt if used_dt is not None else 1.0,
                          n_steps=n_steps, state_names=list(state_cols),
                          theta_names=list(theta_cols), x_max=x_max_arr)

# -----------------------------
# Dictionnaires (phi, psi)
# -----------------------------
def dict_phi_x(x: np.ndarray, degree: int = 2, include_cross: bool = True, include_bias: bool = True) -> np.ndarray:
    n = x.shape[-1]
    feats = [x[..., i] for i in range(n)]
    if degree >= 2:
        feats += [x[..., i]*x[..., i] for i in range(n)]
        if include_cross:
            for i in range(n):
                for j in range(i+1, n):
                    feats.append(x[..., i]*x[..., j])
    if degree >= 3:
        for i in range(n):
            for j in range(n):
                if i != j:
                    feats.append((x[..., i]*x[..., i])*x[..., j])
    if include_bias:
        feats.append(np.ones_like(x[..., 0]))
    return np.stack(feats, axis=-1)

def psi_theta(theta: np.ndarray, include_bias: bool = True, include_cross: bool = False) -> np.ndarray:
    m = theta.shape[-1]
    feats = []
    if include_bias:
        feats.append(np.ones_like(theta[..., 0]))
    feats += [theta[..., i] for i in range(m)]
    if include_cross:
        for i in range(m):
            for j in range(i+1, m):
                feats.append(theta[..., i]*theta[..., j])
    return np.stack(feats, axis=-1)

# -----------------------------
# LPV-EDMD core
# -----------------------------
@dataclass
class LPVKoopman:
    K_blocks: List[np.ndarray]  # (n_phi, n_phi) par bloc psi_j
    phi_fun: Callable[[np.ndarray], np.ndarray]
    psi_fun: Callable[[np.ndarray], np.ndarray]
    n_phi: int
    m_psi: int
    n_states: int
    x_max: Optional[np.ndarray] = None

    def K_of(self, theta: np.ndarray) -> np.ndarray:
        psi = self.psi_fun(theta)
        K = np.zeros((self.n_phi, self.n_phi), dtype=float)
        for j, Kj in enumerate(self.K_blocks):
            K += float(psi[..., j]) * Kj
        return K

    def get_K(self, theta: Union[np.ndarray, Dict[str, float], Sequence[float]],
              order: Optional[Sequence[str]] = None) -> np.ndarray:
        if isinstance(theta, dict):
            if order is None:
                raise ValueError("Fournis 'order' (= ds.theta_names) pour convertir un dict en vecteur.")
            v = np.array([theta[k] for k in order], dtype=float)
        else:
            v = np.asarray(theta, dtype=float)
        return self.K_of(v)

    def predict_one_step(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        phi_x = self.phi_fun(x[None, :])[0]
        phi_next = self.K_of(theta) @ phi_x
        z = phi_next[: self.n_states]
        # clamp physiques
        z = np.clip(z, 0.0, self.x_max if self.x_max is not None else None)
        return z

    def rollout(self, x0: np.ndarray, Theta_seq: np.ndarray) -> np.ndarray:
        H = Theta_seq.shape[0]
        X = np.zeros((H+1, self.n_states), dtype=float)
        X[0] = x0
        for k in range(H):
            X[k+1] = self.predict_one_step(X[k], Theta_seq[k])
        return X

def _build_regression(ds: LPVDatasetFlow, phi_fun, psi_fun):
    Xk = ds.X.reshape(-1, ds.X.shape[-1])
    Xk1 = ds.X_next.reshape(-1, ds.X_next.shape[-1])
    Th = ds.Theta.reshape(-1, ds.Theta.shape[-1])
    Phi_k  = phi_fun(Xk)
    Phi_k1 = phi_fun(Xk1)
    Psi    = psi_fun(Th)
    n_phi = Phi_k.shape[1]
    m = Psi.shape[1]
    W = np.concatenate([Psi[:, j:j+1] * Phi_k for j in range(m)], axis=1)  # (N, m*n_phi)
    Y = Phi_k1
    return W, Y, n_phi, m

def fit_lpv_koopman(ds: LPVDatasetFlow,
                    phi_fun: Callable[[np.ndarray], np.ndarray],
                    psi_fun: Callable[[np.ndarray], np.ndarray],
                    ridge_lambda: float = 1e-6) -> LPVKoopman:
    W, Y, n_phi, m = _build_regression(ds, phi_fun, psi_fun)
    WT_W = W.T @ W
    if ridge_lambda > 0:
        WT_W = WT_W + ridge_lambda * np.eye(WT_W.shape[0])
    A = np.linalg.solve(WT_W, W.T @ Y)
    K_blocks = [(A[j*n_phi:(j+1)*n_phi, :]).T for j in range(m)]
    return LPVKoopman(K_blocks=K_blocks, phi_fun=phi_fun, psi_fun=psi_fun,
                      n_phi=n_phi, m_psi=m, n_states=ds.X.shape[-1], x_max=ds.x_max)

# -----------------------------
# Visualisation / comparaison
# -----------------------------
def plot_rollout_states(t: np.ndarray, X_true: Optional[np.ndarray],
                        X_pred: np.ndarray, state_names: Sequence[str],
                        title: str = "Rollout"):
    fig = go.Figure()
    for i, name in enumerate(state_names):
        if X_true is not None:
            fig.add_trace(go.Scatter(x=t, y=X_true[:, i], mode="lines", name=f"{name} (vrai)"))
            fig.add_trace(go.Scatter(x=t, y=X_pred[:, i], mode="lines", name=f"{name} (pred)", line=dict(dash="dash")))
        else:
            fig.add_trace(go.Scatter(x=t, y=X_pred[:, i], mode="lines", name=f"{name} (Koopman)"))
    fig.update_layout(title=title, xaxis_title="Temps (pas)", yaxis_title="Niveau",
                      template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    fig.show()

def compare_koopman_vs_sim(ds: LPVDatasetFlow, model: LPVKoopman,
                           i_traj: int = 0, k0: int = 0, horizon: Optional[int] = None,
                           title: str = "Rollout — fenêtre d'entraînement"):
    if horizon is None:
        horizon = ds.n_steps - k0
    t = np.arange(horizon+1) * ds.dt
    X_true = np.vstack([ds.X[i_traj, k0], ds.X_next[i_traj, k0:k0+horizon]])
    Theta_seg = ds.Theta[i_traj, k0:k0+horizon]
    X_pred = model.rollout(ds.X[i_traj, k0], Theta_seg)
    plot_rollout_states(t, X_true, X_pred, ds.state_names, title=title)
    rmse = np.sqrt(np.mean((X_true - X_pred)**2, axis=0))
    print("RMSE par état:", {n: float(r) for n, r in zip(ds.state_names, rmse)})

# -----------------------------
# Fit pipeline
# -----------------------------
def fit_from_folder(data_dir: Optional[PathLike] = None,
                    state_cols: Sequence[str] = DEFAULT_STATE_COLS,
                    theta_cols: Sequence[str] = DEFAULT_THETA_COLS,
                    n_steps: int = 600, stride: int = 300,
                    ridge_lambda: float = 1e-6,
                    phi_degree: int = 2,
                    psi_cross: bool = False,
                    x_max: Optional[Sequence[float]] = None) -> tuple[LPVKoopman, LPVDatasetFlow]:
    data = load_flow_csvs(data_dir=data_dir)
    ds = build_lpv_dataset_from_df(data, state_cols=state_cols, theta_cols=theta_cols,
                                   n_steps=n_steps, stride=stride, by_scenario=False, x_max=x_max)
    phi = lambda x: dict_phi_x(x, degree=phi_degree, include_cross=True, include_bias=True)
    psi = lambda th: psi_theta(th, include_bias=True, include_cross=psi_cross)
    model = fit_lpv_koopman(ds, phi, psi, ridge_lambda=ridge_lambda)
    return model, ds




import plotly.graph_objects as go
import plotly.express as px

def plot_3d_true_pred(X_true, X_pred, names=("xA","xB","xC"), title="Trajectoire 3D — vrai vs préd"):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X_true[:,0], y=X_true[:,1], z=X_true[:,2],
                               mode="lines", name="vrai", line=dict(width=5)))
    fig.add_trace(go.Scatter3d(x=X_pred[:,0], y=X_pred[:,1], z=X_pred[:,2],
                               mode="lines", name="pred", line=dict(width=4, dash="dash")))
    fig.update_layout(
        title=title, template="plotly_white",
        scene=dict(xaxis_title=names[0], yaxis_title=names[1], zaxis_title=names[2], aspectmode="cube"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
    )
    fig.show()



# -----------------------------
# main (démo)
# -----------------------------
if __name__ == "__main__":
    # 1) Localiser <racine>/data2
    OUTPUT_DIR = _resolve_data2_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[io] OUTPUT_DIR = {OUTPUT_DIR}")

    # 2) Entraîner Koopman LPV
    model, ds = fit_from_folder(data_dir=OUTPUT_DIR,
                                n_steps=600, stride=300, ridge_lambda=1e-6,
                                phi_degree=2, psi_cross=False)

    print(f"Entraîné: n_states={model.n_states}, n_phi={model.n_phi}, m_psi={model.m_psi}")
    print(f"États: {ds.state_names}")
    print(f"Features θ: {ds.theta_names}")

    # 3) Comparer vrai vs Koopman sur une fenêtre d'entraînement
    compare_koopman_vs_sim(ds, model, i_traj=0, k0=0, horizon=ds.n_steps,
                           title="Rollout — fenêtre d'entraînement (flowline)")
                           

    
    # Exemple d’appel avec ton dataset/modèle déjà entraînés :
    i = 25
    X_true = np.vstack([ds.X[i,0], ds.X_next[i]])          # (H+1, 3)
    Theta_seg = ds.Theta[i]                                 # (H, m)
    X_pred = model.rollout(ds.X[i,0], Theta_seg)            # (H+1, 3)
    plot_3d_true_pred(X_true, X_pred, names=ds.state_names)
