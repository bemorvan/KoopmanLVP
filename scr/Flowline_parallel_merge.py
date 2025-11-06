# flowline_generate_csv_piecewise.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================
# Parallel + Merge flowline — DATA GENERATOR (with PIECEWISE bank)
# - Génère les 4 scénarios déterministes (nominal, m2_slow, m3_setup, m1_quality_drop)
# - Génère N trajectoires "piecewise" aléatoires (banque d'entraînement)
# - Écrit dans <racine>/data2 (jamais dans src/scr)
# =============================================================

# -----------------------------
# Helpers
# -----------------------------
def _resolve_data2_root() -> Path:
    """Trouve/Crée <racine>/data2 en partant du CWD (et de ses parents).
    Si absent, le crée dans le CWD (considéré comme racine du projet)."""
    cwd = Path.cwd()
    # 1) existe déjà dans CWD ou parents ?
    for base in [cwd, *cwd.parents]:
        d = base / "data2"
        if d.is_dir():
            return d.resolve()
    # 2) si on exécute depuis src|scr, remonter d'un cran
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = cwd
    base_guess = script_dir.parent if script_dir.name.lower() in {"src", "scr"} else cwd
    d = (base_guess / "data2").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d

def logistic(z: float, k: float) -> float:
    return 1.0 / (1.0 + np.exp(-k * z))

def softmax2(a: float, b: float, kappa: float) -> Tuple[float, float]:
    wa = np.exp(kappa * a); wb = np.exp(kappa * b)
    s = wa + wb
    return float(wa / s), float(wb / s)

def sample_piecewise_schedule(rng: np.random.Generator, T: int,
                              value_range: Tuple[float, float],
                              seg_len_range: Tuple[int, int] = (60, 300),
                              noise_std: float = 0.0) -> np.ndarray:
    vals: List[float] = []
    remaining = T
    while remaining > 0:
        L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
        L = min(L, remaining)
        v = float(rng.uniform(*value_range))
        vals.extend([v] * L)
        remaining -= L
    arr = np.asarray(vals, dtype=float)
    if noise_std > 0:
        arr = np.clip(arr + rng.normal(0.0, noise_std, size=arr.shape), value_range[0], value_range[1])
    return arr

def sample_piecewise_binary(rng: np.random.Generator, T: int,
                            p_down: float = 0.15,
                            seg_len_range: Tuple[int, int] = (30, 180)) -> np.ndarray:
    vals: List[float] = []
    remaining = T
    cur = 1.0
    while remaining > 0:
        L = int(rng.integers(seg_len_range[0], seg_len_range[1] + 1))
        L = min(L, remaining)
        if cur > 0.5:
            cur = 0.0 if rng.random() < p_down else 1.0
        else:
            cur = 1.0
        vals.extend([cur] * L)
        remaining -= L
    return np.asarray(vals, dtype=float)

# -----------------------------
# Simulator config & profiles
# -----------------------------
@dataclass
class LineConfig:
    BA: float = 40.0
    BB: float = 40.0
    BC: float = 60.0
    k_gate: float = 3.0
    eps_gate: float = 0.5
    kappa_pull: float = 0.12
    dt: float = 1.0
    # measurement noise
    noise_x: float = 0.05
    noise_s: float = 0.02

@dataclass
class Profiles:
    nu1: np.ndarray; nu2: np.ndarray; nu3: np.ndarray
    a1: np.ndarray;  a2: np.ndarray;  a3: np.ndarray
    q1: np.ndarray;  q2: np.ndarray;  q3: np.ndarray
    lambda1_in: np.ndarray; lambda2_in: np.ndarray
    tags: Dict[str, np.ndarray]

class ParallelMergeSimulator:
    def __init__(self, cfg: LineConfig):
        self.cfg = cfg

    # ----- step -----
    def step(self, x: np.ndarray, pars: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        BA, BB, BC = self.cfg.BA, self.cfg.BB, self.cfg.BC
        k, eps, kap = self.cfg.k_gate, self.cfg.eps_gate, self.cfg.kappa_pull
        dt = self.cfg.dt
        xA, xB, xC = map(float, x)

        # upstream machines
        g_block_A = logistic(BA - xA - eps, k)
        g_block_B = logistic(BB - xB - eps, k)
        s1_out = pars['nu1'] * pars['a1'] * pars['q1'] * g_block_A
        s2_out = pars['nu2'] * pars['a2'] * pars['q2'] * g_block_B

        # merge machine
        wA, wB = softmax2(xA, xB, kap)
        g_pull_A = logistic(xA - eps, k)
        g_pull_B = logistic(xB - eps, k)
        g_down   = logistic(BC - xC - eps, k)
        base_pull = pars['nu3'] * pars['a3'] * pars['q3'] * g_down
        fA = min(wA * base_pull * g_pull_A, xA / dt)
        fB = min(wB * base_pull * g_pull_B, xB / dt)

        s3_out = pars['nu3'] * pars['a3'] * pars['q3'] * logistic(xC - eps, k)

        # buffers dyn.
        dxA = pars['lambda1_in'] - s1_out - fA
        dxB = pars['lambda2_in'] - s2_out - fB
        dxC = fA + fB - s3_out
        x_next = np.array([
            np.clip(xA + dt * dxA, 0.0, BA),
            np.clip(xB + dt * dxB, 0.0, BB),
            np.clip(xC + dt * dxC, 0.0, BC)
        ], dtype=float)

        outs = {
            's1_out': s1_out, 's2_out': s2_out, 's3_out': s3_out,
            'fA_pull': fA, 'fB_pull': fB,
            'wA': wA, 'wB': wB,
            'g_block_A': g_block_A, 'g_block_B': g_block_B,
            'g_pull_A': g_pull_A, 'g_pull_B': g_pull_B, 'g_down': g_down,
        }
        return x_next, outs

    # ----- deterministic profiles -----
    def build_profiles(self, T: int, scenario: str = 'nominal') -> Profiles:
        dt = self.cfg.dt
        t = np.arange(T) * dt
        nu1 = np.full(T, 1.2); nu2 = np.full(T, 1.0); nu3 = np.full(T, 1.8)
        a1 = np.ones(T); a2 = np.ones(T); a3 = np.ones(T)
        q1 = np.full(T, 0.99); q2 = np.full(T, 0.99); q3 = np.ones(T)
        lambda1_in = 0.9 * nu1 * (1.0 + 0.1 * np.sin(2*np.pi*t/900))
        lambda2_in = 0.9 * nu2 * (1.0 + 0.1 * np.sin(2*np.pi*t/1100 + 1.1))
        tags = {'m3_setup': np.zeros(T), 'fault_m2': np.zeros(T)}

        if scenario == 'm2_slow':
            start = int(20*60/dt); end = int(50*60/dt)
            nu2[start:end] *= 0.8
            for s in [int(25*60/dt), int(40*60/dt), int(65*60/dt)]:
                e = s + int(2.5*60/dt)
                a2[s:e] = 0.0
            tags['fault_m2'][a2 < 0.5] = 1.0
        elif scenario == 'm3_setup':
            s = int(40*60/dt); e = s + int(8*60/dt)
            a3[s:e] = 0.0
            tags['m3_setup'][s:e] = 1.0
        elif scenario == 'm1_quality_drop':
            s = int(55*60/dt); e = s + int(15*60/dt)
            q1[s:e] = 0.90

        return Profiles(nu1,nu2,nu3,a1,a2,a3,q1,q2,q3,lambda1_in,lambda2_in,tags)

    # ----- piecewise random profiles -----
    def build_profiles_piecewise(self, T: int, rng: np.random.Generator,
                                 seg_len_range: Tuple[int, int] = (60, 300),
                                 ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Profiles:
        if ranges is None:
            ranges = {
                'nu1': (0.8, 1.8), 'nu2': (0.8, 1.6), 'nu3': (1.2, 2.4),
                'q1': (0.92, 1.0), 'q2': (0.92, 1.0), 'q3': (0.95, 1.0),
                'lambda1_factor': (0.65, 0.95), 'lambda2_factor': (0.65, 0.95),
            }
        nu1 = sample_piecewise_schedule(rng, T, ranges['nu1'], seg_len_range, noise_std=0.02)
        nu2 = sample_piecewise_schedule(rng, T, ranges['nu2'], seg_len_range, noise_std=0.02)
        nu3 = sample_piecewise_schedule(rng, T, ranges['nu3'], seg_len_range, noise_std=0.02)
        q1  = sample_piecewise_schedule(rng, T, ranges['q1'], seg_len_range)
        q2  = sample_piecewise_schedule(rng, T, ranges['q2'], seg_len_range)
        q3  = sample_piecewise_schedule(rng, T, ranges['q3'], seg_len_range)
        a1 = sample_piecewise_binary(rng, T, p_down=0.10, seg_len_range=(30, 120))
        a2 = sample_piecewise_binary(rng, T, p_down=0.18, seg_len_range=(30, 120))
        a3 = sample_piecewise_binary(rng, T, p_down=0.12, seg_len_range=(30, 120))
        lam1_f = sample_piecewise_schedule(rng, T, ranges['lambda1_factor'], seg_len_range)
        lam2_f = sample_piecewise_schedule(rng, T, ranges['lambda2_factor'], seg_len_range)
        lambda1_in = lam1_f * nu1
        lambda2_in = lam2_f * nu2
        tags = {'m3_setup': (a3 < 0.5).astype(float), 'fault_m2': (a2 < 0.5).astype(float)}
        return Profiles(nu1,nu2,nu3,a1,a2,a3,q1,q2,q3,lambda1_in,lambda2_in,tags)

    # ----- simulate generic -----
    def simulate_from_profiles(self, T: int, prof: Profiles,
                               x0: Optional[Sequence[float]] = None,
                               add_noise: bool = True,
                               seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = self.cfg.dt
        x = np.array([10.0, 10.0, 15.0], dtype=float) if x0 is None else np.array(x0, dtype=float)

        rows: List[Dict[str, float]] = []
        n1_cum = n2_cum = n3_cum = 0.0
        for k in range(T):
            pars = {
                'nu1': float(prof.nu1[k]), 'nu2': float(prof.nu2[k]), 'nu3': float(prof.nu3[k]),
                'a1': float(prof.a1[k]),   'a2': float(prof.a2[k]),   'a3': float(prof.a3[k]),
                'q1': float(prof.q1[k]),   'q2': float(prof.q2[k]),   'q3': float(prof.q3[k]),
                'lambda1_in': float(prof.lambda1_in[k]),
                'lambda2_in': float(prof.lambda2_in[k]),
            }
            x_next, outs = self.step(x, pars)
            n1_cum += outs['s1_out'] * dt
            n2_cum += outs['s2_out'] * dt
            n3_cum += outs['s3_out'] * dt

            if add_noise:
                xA_meas = x[0] + rng.normal(0.0, self.cfg.noise_x)
                xB_meas = x[1] + rng.normal(0.0, self.cfg.noise_x)
                xC_meas = x[2] + rng.normal(0.0, self.cfg.noise_x)
                s1_meas = outs['s1_out'] + rng.normal(0.0, self.cfg.noise_s)
                s2_meas = outs['s2_out'] + rng.normal(0.0, self.cfg.noise_s)
                s3_meas = outs['s3_out'] + rng.normal(0.0, self.cfg.noise_s)
            else:
                xA_meas, xB_meas, xC_meas = x[0], x[1], x[2]
                s1_meas, s2_meas, s3_meas = outs['s1_out'], outs['s2_out'], outs['s3_out']

            rows.append({
                'time_s': k*dt, 'scenario': 'piecewise',
                'xA': xA_meas, 'xB': xB_meas, 'xC': xC_meas,
                's1_out': s1_meas, 's2_out': s2_meas, 's3_out': s3_meas,
                'n1_out_cum': n1_cum, 'n2_out_cum': n2_cum, 'n3_out_cum': n3_cum,
                'nu1': pars['nu1'], 'nu2': pars['nu2'], 'nu3': pars['nu3'],
                'a1': pars['a1'], 'a2': pars['a2'], 'a3': pars['a3'],
                'q1': pars['q1'], 'q2': pars['q2'], 'q3': pars['q3'],
                'lambda1_in': pars['lambda1_in'], 'lambda2_in': pars['lambda2_in'],
                'm3_setup': float(prof.tags['m3_setup'][k]), 'fault_m2': float(prof.tags['fault_m2'][k]),
                'fA_pull': outs['fA_pull'], 'fB_pull': outs['fB_pull'],
                'wA': outs['wA'], 'wB': outs['wB'],
                'g_block_A': outs['g_block_A'], 'g_block_B': outs['g_block_B'],
                'g_pull_A': outs['g_pull_A'], 'g_pull_B': outs['g_pull_B'], 'g_down': outs['g_down'],
            })
            x = x_next
        return pd.DataFrame(rows)

    def simulate(self, T: int, scenario: str, x0: Optional[Sequence[float]] = None,
                 add_noise: bool = True, seed: Optional[int] = None) -> pd.DataFrame:
        prof = self.build_profiles(T, scenario)
        return self.simulate_from_profiles(T, prof, x0=x0, add_noise=add_noise, seed=seed)

    # ----- CSV writers -----
    def simulate_scenarios_to_csv(self, out_dir: Path, duration_s: int,
                                  scenarios: Optional[List[str]] = None,
                                  seed: Optional[int] = 123) -> List[Path]:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[generator] Writing CSVs to: {out_dir}")
        if scenarios is None:
            scenarios = ['nominal', 'm2_slow', 'm3_setup', 'm1_quality_drop']
        csv_paths: List[Path] = []
        for sc in scenarios:
            df = self.simulate(T=int(duration_s/self.cfg.dt), scenario=sc, add_noise=True, seed=seed)
            path = out_dir / f"flowline_{sc}.csv"
            df.to_csv(path, index=False)
            print(f"  - saved {path}  [rows={len(df)}]")
            csv_paths.append(path)
        df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
        combo = out_dir / "flowline_all_scenarios.csv"
        df_all.to_csv(combo, index=False)
        print(f"  - saved {combo}  [rows={len(df_all)}]")
        csv_paths.append(combo)
        return csv_paths

    def simulate_piecewise_bank_to_csv(self, out_dir: Path, duration_s: int,
                                       n_runs: int = 40,
                                       seg_len_range: Tuple[int,int] = (60, 300),
                                       ranges: Optional[Dict[str, Tuple[float,float]]] = None,
                                       seed: Optional[int] = 123) -> List[Path]:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        print(f"[generator] Writing PIECEWISE bank to: {out_dir} (n_runs={n_runs})")
        paths: List[Path] = []
        T = int(duration_s / self.cfg.dt)
        for i in range(n_runs):
            prof = self.build_profiles_piecewise(T, rng, seg_len_range=seg_len_range, ranges=ranges)
            df = self.simulate_from_profiles(T, prof, add_noise=True, seed=rng.integers(1<<30))
            path = out_dir / f"flowline_rand_{i:03d}.csv"
            df.to_csv(path, index=False)
            print(f"  - saved {path}  [rows={len(df)}]")
            paths.append(path)
        combo = out_dir / "flowline_piecewise_bank.csv"
        pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(combo, index=False)
        print(f"  - saved {combo}")
        paths.append(combo)
        return paths

# -----------------------------
# main
# -----------------------------
if __name__ == '__main__':
    # 1) Paramètres
    DURATION_MIN = 30.0
    DT = 1.0
    NOISE_X = 0.00  # 0.05 si tu veux du bruit
    NOISE_S = 0.00  # 0.02 si tu veux du bruit
    SEED = 123

    # 2) Où écrire ? -> <racine>/data2
    OUTPUT_DIR = _resolve_data2_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[io] OUTPUT_DIR = {OUTPUT_DIR}")

    # 3) Génération
    GENERATE_DETERMINISTIC = True
    GENERATE_PIECEWISE_BANK = True
    N_RANDOM = 60                # plus de diversité
    SEG_LEN_RANGE = (60, 300)

    cfg = LineConfig(dt=DT, noise_x=NOISE_X, noise_s=NOISE_S)
    sim = ParallelMergeSimulator(cfg)

    all_paths: List[Path] = []
    if GENERATE_DETERMINISTIC:
        all_paths += sim.simulate_scenarios_to_csv(out_dir=OUTPUT_DIR, duration_s=int(DURATION_MIN*60), seed=SEED)
    if GENERATE_PIECEWISE_BANK and N_RANDOM > 0:
        all_paths += sim.simulate_piecewise_bank_to_csv(out_dir=OUTPUT_DIR, duration_s=int(DURATION_MIN*60),
                                                        n_runs=N_RANDOM, seg_len_range=SEG_LEN_RANGE, seed=SEED)

    print("\nGenerated:")
    for p in all_paths:
        print("  ", p)
