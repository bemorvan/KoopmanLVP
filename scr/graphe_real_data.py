from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Ouvre dans le navigateur (VS Code friendly)
pio.renderers.default = "browser"

PathLike = Union[str, Path]

# -----------------------------------------------------
# Chargement utilitaires
# -----------------------------------------------------

def _paths(items: Iterable[PathLike]) -> List[Path]:
    return [Path(p).expanduser().resolve() for p in items]


def load_csv(path: PathLike) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    if 'time_s' in df.columns:
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    if 'scenario' not in df.columns:
        df['scenario'] = p.stem.replace('flowline_', '')
    return df


def list_runs(data_dir: PathLike) -> List[Path]:
    d = Path(data_dir).expanduser().resolve()
    if not d.exists():
        return []
    # prioriser nominal / variations / puis banque aléatoire
    names = [
        'flowline_nominal.csv', 'flowline_m2_slow.csv',
        'flowline_m3_setup.csv', 'flowline_m1_quality_drop.csv'
    ]
    ordered: List[Path] = [p for p in (d / n for n in names) if p.exists()]
    ordered += sorted(d.glob('flowline_rand_*.csv'))
    if (d / 'flowline_piecewise_bank.csv').exists():
        ordered.append(d / 'flowline_piecewise_bank.csv')
    return ordered


# -----------------------------------------------------
# Plots multi-trajectoires
# -----------------------------------------------------

def plot_3d_buffers(paths: Iterable[PathLike], labels: Optional[List[str]] = None, downsample: int = 1):
    """Trace xA-xB-xC en 3D pour plusieurs fichiers CSV."""
    pts = _paths(paths)
    fig = go.Figure()
    pal = px.colors.qualitative.Plotly
    for j, p in enumerate(pts):
        d = load_csv(p)
        d = d.iloc[::max(1, downsample)]
        name = labels[j] if labels and j < len(labels) else p.stem.replace('flowline_', '')
        fig.add_trace(go.Scatter3d(
            x=d['xA'], y=d['xB'], z=d['xC'], mode='lines',
            name=name, line=dict(width=4, color=pal[j % len(pal)])
        ))
    fig.update_layout(
        title='Trajectoires 3D — buffers (xA, xB, xC)', template='plotly_white',
        scene=dict(xaxis_title='xA', yaxis_title='xB', zaxis_title='xC', aspectmode='cube'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


def plot_overlay(paths: Iterable[PathLike], column: str, title: Optional[str] = None, downsample: int = 1):
    """Superpose une colonne (ex: 'xC' ou 's3_out') sur plusieurs fichiers."""
    pts = _paths(paths)
    fig = go.Figure()
    pal = px.colors.qualitative.Plotly
    for j, p in enumerate(pts):
        d = load_csv(p)
        if column not in d.columns:
            raise ValueError(f"Colonne '{column}' absente dans {p.name}")
        d = d.iloc[::max(1, downsample)]
        name = p.stem.replace('flowline_', '')
        fig.add_trace(go.Scatter(x=d['time_s'], y=d[column], mode='lines',
                                 name=name, line=dict(color=pal[j % len(pal)])))
    fig.update_layout(
        title=title or f"Overlay — {column}", template='plotly_white',
        xaxis_title='Temps (s)', yaxis_title=column,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


def plot_params(paths: Iterable[PathLike], params: List[str] = ['nu1','nu2','nu3','a1','a2','a3'], downsample: int = 1):
    """Petit tableau de paramètres (en 2 lignes) superposés par trajectoire."""
    pts = _paths(paths)
    pal = px.colors.qualitative.Plotly
    # deux figures empilées pour ne pas surcharger
    top, bottom = params[:len(params)//2], params[len(params)//2:]
    for group, title in [(top, 'Paramètres (groupe 1)'), (bottom, 'Paramètres (groupe 2)')]:
        if not group:
            continue
        fig = go.Figure()
        for j, p in enumerate(pts):
            d = load_csv(p).iloc[::max(1, downsample)]
            name = p.stem.replace('flowline_', '')
            for param in group:
                if param not in d.columns:
                    continue
                fig.add_trace(go.Scatter(x=d['time_s'], y=d[param], mode='lines',
                                         name=f"{name} • {param}",
                                         line=dict(color=pal[j % len(pal)]),
                                         legendgroup=name, showlegend=True))
        fig.update_layout(
            title=title, template='plotly_white',
            xaxis_title='Temps (s)', yaxis_title='valeur',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
        )
        fig.show()


# -----------------------------------------------------
# Démo (▶️ Run)
# -----------------------------------------------------
if __name__ == '__main__':
    # 1) Où écrire / lire
    script_dir = Path(__file__).resolve().parent
    data_dir  = 'data2'
    OUTPUT_DIR = Path(data_dir).expanduser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Choisir quelques trajectoires à comparer
    runs = list_runs(OUTPUT_DIR)
    if not runs:
        print(f"Aucun CSV dans {OUTPUT_DIR}. Lances d'abord le générateur.")
        raise SystemExit(0)

    # Exemple: prends 4 fichiers (nominal, m2_slow, m3_setup, et un rand si dispo)
    chosen: List[Path] = ["flowline_rand_001", "flowline_rand_002", "flowline_rand_003"]
    for base in ['flowline_nominal.csv','flowline_m2_slow.csv','flowline_m3_setup.csv']:
        p = OUTPUT_DIR / base
        if p.exists():
            chosen.append(p)
    #ajoute un aléatoire s'il existe
    rands = [p for p in runs if p.name.startswith('flowline_rand_')]
    if rands:
        chosen.append(rands[0])

    #sinon, prends les 3-4 premiers disponibles
    if len(chosen) < 3:
        extra = [p for p in runs if p not in chosen][: (4 - len(chosen))]
        chosen += extra

    print("Comparaison sur:")
    for p in chosen:
        print("  -", p.name)

    # 3) Tracés
    plot_3d_buffers(chosen, downsample=2)
    plot_overlay(chosen, column='xC', title='Buffer xC — multi-trajectoires', downsample=1)
    plot_overlay(chosen, column='s3_out', title='Débit s3_out — multi-trajectoires', downsample=1)
    plot_params(chosen, params=['nu1','nu2','nu3','a1','a2','a3'], downsample=5)
