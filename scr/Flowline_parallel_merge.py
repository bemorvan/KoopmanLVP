from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================
# Parallel + Merge flowline — DATA GENERATOR ONLY
# Generates CSV files with sensor-like measurements.
# Default output directory is "\\data" (Windows-root style). Change with --out.
# =============================================================

# -----------------------------
# Helpers
# -----------------------------

def logistic(z: float, k: float) -> float:
    """Smooth gate in (0,1) used for blocking/starvation and capacity limits."""
    return 1.0 / (1.0 + np.exp(-k * z))


def softmax2(a: float, b: float, kappa: float) -> Tuple[float, float]:
    """2-term softmax used to bias the merge pulling towards the more filled buffer."""
    wa = np.exp(kappa * a)
    wb = np.exp(kappa * b)
    s = wa + wb
    return float(wa / s), float(wb / s)


# -----------------------------
# Simulator config & profiles
# -----------------------------
@dataclass
class LineConfig:
    BA: float = 40.0  # capacity buffer A (pieces)
    BB: float = 40.0  # capacity buffer B (pieces)
    BC: float = 60.0  # capacity buffer C (pieces)
    k_gate: float = 3.0   # steepness for logistic gates
    eps_gate: float = 0.5 # margin for gates
    kappa_pull: float = 0.12  # softmax bias for M3 pulling
    dt: float = 1.0  # seconds per step
    # measurement noise (std)
    noise_x: float = 0.05   # pieces
    noise_s: float = 0.02   # pieces/second


@dataclass
class Profiles:
    """Time-series for scheduling and inputs (length T)."""
    nu1: np.ndarray; nu2: np.ndarray; nu3: np.ndarray
    a1: np.ndarray;  a2: np.ndarray;  a3: np.ndarray
    q1: np.ndarray;  q2: np.ndarray;  q3: np.ndarray
    lambda1_in: np.ndarray; lambda2_in: np.ndarray
    tags: Dict[str, np.ndarray]


class ParallelMergeSimulator:
    def __init__(self, cfg: LineConfig):
        self.cfg = cfg

    def step(self, x: np.ndarray, pars: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """One forward-Euler step. x = [xA,xB,xC].
        pars: nu1,nu2,nu3,a1,a2,a3,q1,q2,q3,lambda1_in,lambda2_in
        Returns (x_next, outputs_dict).
        """
        BA, BB, BC = self.cfg.BA, self.cfg.BB, self.cfg.BC
        k, eps, kap = self.cfg.k_gate, self.cfg.eps_gate, self.cfg.kappa_pull
        dt = self.cfg.dt
        xA, xB, xC = map(float, x)

        # Upstream machines (blocking soft on full buffers)
        g_block_A = logistic(BA - xA - eps, k)
        g_block_B = logistic(BB - xB - eps, k)
        s1_out = pars['nu1'] * pars['a1'] * pars['q1'] * g_block_A
        s2_out = pars['nu2'] * pars['a2'] * pars['q2'] * g_block_B

        # Merge machine M3 pulls from A and B (softmax bias + starvation + downstream room)
        wA, wB = softmax2(xA, xB, kap)
        g_pull_A = logistic(xA - eps, k)
        g_pull_B = logistic(xB - eps, k)
        g_down   = logistic(BC - xC - eps, k)
        base_pull = pars['nu3'] * pars['a3'] * pars['q3'] * g_down
        fA = wA * base_pull * g_pull_A
        fB = wB * base_pull * g_pull_B

        # cannot pull more than available this step
        fA = min(fA, xA / dt)
        fB = min(fB, xB / dt)

        # M3 output to downstream (drain buffer C)
        s3_out = pars['nu3'] * pars['a3'] * pars['q3'] * logistic(xC - eps, k)

        # Buffer dynamics
        dxA = pars['lambda1_in'] - s1_out - fA
        dxB = pars['lambda2_in'] - s2_out - fB
        dxC = fA + fB - s3_out
        xA_next = np.clip(xA + dt * dxA, 0.0, BA)
        xB_next = np.clip(xB + dt * dxB, 0.0, BB)
        xC_next = np.clip(xC + dt * dxC, 0.0, BC)
        x_next = np.array([xA_next, xB_next, xC_next], dtype=float)

        outs = {
            's1_out': s1_out,
            's2_out': s2_out,
            's3_out': s3_out,
            'fA_pull': fA,
            'fB_pull': fB,
            'wA': wA,
            'wB': wB,
            'g_block_A': g_block_A,
            'g_block_B': g_block_B,
            'g_pull_A': g_pull_A,
            'g_pull_B': g_pull_B,
            'g_down': g_down,
        }
        return x_next, outs

    # ---------- Scenario profiles ----------
    def build_profiles(self, T: int, scenario: str = 'nominal') -> Profiles:
        dt = self.cfg.dt
        t = np.arange(T) * dt
        # Base profiles
        nu1 = np.full(T, 1.2); nu2 = np.full(T, 1.0); nu3 = np.full(T, 1.8)
        a1 = np.ones(T); a2 = np.ones(T); a3 = np.ones(T)
        q1 = np.full(T, 0.99); q2 = np.full(T, 0.99); q3 = np.ones(T)
        # upstream launches (slow sin ±10%)
        lambda1_in = 0.9 * nu1 * (1.0 + 0.1 * np.sin(2*np.pi*t/900))
        lambda2_in = 0.9 * nu2 * (1.0 + 0.1 * np.sin(2*np.pi*t/1100 + 1.1))
        tags = {'m3_setup': np.zeros(T), 'fault_m2': np.zeros(T)}

        if scenario == 'nominal':
            pass
        elif scenario == 'm2_slow':
            # Reduce nu2 between 20–50 min + micro stops
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
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return Profiles(nu1,nu2,nu3,a1,a2,a3,q1,q2,q3,lambda1_in,lambda2_in,tags)

    # ---------- Core simulation ----------
    def simulate(self, T: int, scenario: str, x0: Optional[Sequence[float]] = None,
                 add_noise: bool = True, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cfg = self.cfg
        prof = self.build_profiles(T, scenario)
        dt = cfg.dt
        # initial buffers
        x = np.array([10.0, 10.0, 15.0], dtype=float) if x0 is None else np.array(x0, dtype=float)

        rows: List[Dict[str, float]] = []
        n1_cum = 0.0; n2_cum = 0.0; n3_cum = 0.0
        for k in range(T):
            pars = {
                'nu1': float(prof.nu1[k]), 'nu2': float(prof.nu2[k]), 'nu3': float(prof.nu3[k]),
                'a1': float(prof.a1[k]),   'a2': float(prof.a2[k]),   'a3': float(prof.a3[k]),
                'q1': float(prof.q1[k]),   'q2': float(prof.q2[k]),   'q3': float(prof.q3[k]),
                'lambda1_in': float(prof.lambda1_in[k]),
                'lambda2_in': float(prof.lambda2_in[k]),
            }
            x_next, outs = self.step(x, pars)

            # counters (good pieces)
            n1_cum += outs['s1_out'] * dt
            n2_cum += outs['s2_out'] * dt
            n3_cum += outs['s3_out'] * dt

            # measurement noise
            if add_noise:
                xA_meas = x[0] + rng.normal(0.0, cfg.noise_x)
                xB_meas = x[1] + rng.normal(0.0, cfg.noise_x)
                xC_meas = x[2] + rng.normal(0.0, cfg.noise_x)
                s1_meas = outs['s1_out'] + rng.normal(0.0, cfg.noise_s)
                s2_meas = outs['s2_out'] + rng.normal(0.0, cfg.noise_s)
                s3_meas = outs['s3_out'] + rng.normal(0.0, cfg.noise_s)
            else:
                xA_meas, xB_meas, xC_meas = x[0], x[1], x[2]
                s1_meas, s2_meas, s3_meas = outs['s1_out'], outs['s2_out'], outs['s3_out']

            rows.append({
                'time_s': k*dt,
                'scenario': scenario,
                'xA': xA_meas, 'xB': xB_meas, 'xC': xC_meas,
                's1_out': s1_meas, 's2_out': s2_meas, 's3_out': s3_meas,
                'n1_out_cum': n1_cum, 'n2_out_cum': n2_cum, 'n3_out_cum': n3_cum,
                'nu1': pars['nu1'], 'nu2': pars['nu2'], 'nu3': pars['nu3'],
                'a1': pars['a1'], 'a2': pars['a2'], 'a3': pars['a3'],
                'q1': pars['q1'], 'q2': pars['q2'], 'q3': pars['q3'],
                'lambda1_in': pars['lambda1_in'], 'lambda2_in': pars['lambda2_in'],
                'm3_setup': float(prof.tags['m3_setup'][k]), 'fault_m2': float(prof.tags['fault_m2'][k]),
                # diagnostics (not needed for training, but useful)
                'fA_pull': outs['fA_pull'], 'fB_pull': outs['fB_pull'],
                'wA': outs['wA'], 'wB': outs['wB'],
                'g_block_A': outs['g_block_A'], 'g_block_B': outs['g_block_B'],
                'g_pull_A': outs['g_pull_A'], 'g_pull_B': outs['g_pull_B'], 'g_down': outs['g_down'],
            })
            x = x_next

        return pd.DataFrame(rows)

    def simulate_scenarios_to_csv(
        self,
        out_dir: str | None = None,                 # <- None = par défaut à côté du script
        duration_s: int = 30*60,
        scenarios: Optional[List[str]] = None,
        noise_meas: Tuple[float, float] = (0.05, 0.02),
        seed: Optional[int] = 123
    ) -> List[str]:
        
        
        
        script_dir = Path(__file__).resolve().parent
        out_dir_path = Path(out_dir).expanduser()
        
        out_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[generator] Writing CSVs to: {out_dir_path}")


        # 2) Scénarios
        if scenarios is None:
            scenarios = ['nominal', 'm2_slow', 'm3_setup', 'm1_quality_drop']
        self.cfg.noise_x, self.cfg.noise_s = noise_meas

        # 3) Boucle de génération
        csv_paths: List[str] = []
        for sc in scenarios:
            df = self.simulate(T=int(duration_s/self.cfg.dt), scenario=sc, add_noise=True, seed=seed)
            path = out_dir_path / f"flowline_{sc}.csv"
            df.to_csv(path, index=False)
            print(f"  - saved {path}  [rows={len(df)}]")
            csv_paths.append(str(path))

        # 4) Fichier combiné (optionnel)
        df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
        combo = out_dir_path / "flowline_all_scenarios.csv"
        df_all.to_csv(combo, index=False)
        print(f"  - saved {combo}  [rows={len(df_all)}]")
        csv_paths.append(str(combo))

        return csv_paths



# -----------------------------
# CLI (generation only)
# -----------------------------

def main():
    import os

    # 1) Paramètres faciles à modifier
    DURATION_MIN = 30.0         # durée par scénario (minutes)
    DT = 1.0                    # pas d'échantillonnage (secondes)
    SCENARIOS = ['nominal', 'm2_slow', 'm3_setup', 'm1_quality_drop']
    NOISE_X = 0.00              # bruit capteur sur niveaux (pièces) 0.05
    NOISE_S = 0.00              # bruit capteur sur débits (pièces/s) 0.02
    SEED = 123

    out_dir = "data"

    # 3) Lancer la génération
    cfg = LineConfig(dt=DT, noise_x=NOISE_X, noise_s=NOISE_S)
    sim = ParallelMergeSimulator(cfg)
    paths = sim.simulate_scenarios_to_csv(
        out_dir=out_dir,
        duration_s=int(DURATION_MIN * 60),
        scenarios=SCENARIOS,
        noise_meas=(NOISE_X, NOISE_S),
        seed=SEED
    )

    print("\nGenerated:")
    for pth in paths:
        print("  ", pth)


if __name__ == '__main__':
    main()
