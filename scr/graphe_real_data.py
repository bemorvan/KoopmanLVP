from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Affichage dans le navigateur (VS Code/Windows friendly)
pio.renderers.default = "browser"

# -----------------------------------------------------
# Chargement utilitaire
# -----------------------------------------------------
PathLike = Union[str, Path]


def _to_paths(items: Iterable[PathLike]) -> List[Path]:
    return [Path(p).expanduser().resolve() for p in items]


def load_csv(path: PathLike) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)
    # Sécurité types
    if 'time_s' in df.columns:
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    if 'scenario' in df.columns and df['scenario'].dtype != 'object':
        df['scenario'] = df['scenario'].astype(str)
    return df


def load_many(paths: Iterable[PathLike]) -> pd.DataFrame:
    dfs = []
    for p in _to_paths(paths):
        df = load_csv(p)
        if 'scenario' not in df.columns:
            # inférer depuis le nom de fichier
            df['scenario'] = p.stem.replace('flowline_', '')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# -----------------------------------------------------
# Plots de base par scénario
# -----------------------------------------------------

def plot_buffers(df: pd.DataFrame, title: Optional[str] = None, downsample: int = 1):
    """Trace xA, xB, xC sur le temps pour un scénario.
    downsample: prendre 1 point sur N pour accélérer l'affichage.
    """
    d = df.iloc[::max(1, int(downsample))].copy()
    name = title or (str(d.get('scenario', [''])[0]) or 'buffers')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['xA'], mode='lines', name='xA'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['xB'], mode='lines', name='xB'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['xC'], mode='lines', name='xC'))
    fig.update_layout(
        title=title or f"Buffers (xA,xB,xC) — {name}",
        xaxis_title='Temps (s)', yaxis_title='Pièces',
        template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


def plot_flows(df: pd.DataFrame, title: Optional[str] = None, downsample: int = 1):
    d = df.iloc[::max(1, int(downsample))].copy()
    name = title or (str(d.get('scenario', [''])[0]) or 'flows')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['s1_out'], mode='lines', name='s1_out'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['s2_out'], mode='lines', name='s2_out'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['s3_out'], mode='lines', name='s3_out'))
    fig.update_layout(
        title=title or f"Débits (s1,s2,s3) — {name}",
        xaxis_title='Temps (s)', yaxis_title='Pièces/s',
        template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


def plot_counters(df: pd.DataFrame, title: Optional[str] = None, downsample: int = 1):
    d = df.iloc[::max(1, int(downsample))].copy()
    name = title or (str(d.get('scenario', [''])[0]) or 'counters')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['n1_out_cum'], mode='lines', name='n1_out_cum'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['n2_out_cum'], mode='lines', name='n2_out_cum'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['n3_out_cum'], mode='lines', name='n3_out_cum'))
    fig.update_layout(
        title=title or f"Compteurs cumulés — {name}",
        xaxis_title='Temps (s)', yaxis_title='Pièces cumulées',
        template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


# -----------------------------------------------------
# Overlays multi-scénarios
# -----------------------------------------------------

def plot_throughput_overlay(paths: Iterable[PathLike] | pd.DataFrame, downsample: int = 1):
    """Superpose s3_out pour plusieurs scénarios.
    Accepte soit une liste de chemins CSV, soit un DataFrame combiné avec colonne 'scenario'.
    """
    if isinstance(paths, pd.DataFrame):
        data = paths.copy()
    else:
        data = load_many(paths)

    data = data.iloc[::max(1, int(downsample))]
    fig = go.Figure()
    for scen, d in data.groupby('scenario'):
        fig.add_trace(go.Scatter(x=d['time_s'], y=d['s3_out'], mode='lines', name=f"{scen}"))
    fig.update_layout(
        title='Débit de sortie s3_out — comparaison scénarios',
        xaxis_title='Temps (s)', yaxis_title='Pièces/s', template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    fig.show()


def plot_buffers_overlay(paths: Iterable[PathLike] | pd.DataFrame, buffer_name: str = 'xC', downsample: int = 1):
    """Overlay d'un buffer (xA/xB/xC) sur plusieurs scénarios."""
    assert buffer_name in {'xA','xB','xC'}
    if isinstance(paths, pd.DataFrame):
        data = paths.copy()
    else:
        data = load_many(paths)

    data = data.iloc[::max(1, int(downsample))]
    fig = go.Figure()
    for scen, d in data.groupby('scenario'):
        fig.add_trace(go.Scatter(x=d['time_s'], y=d[buffer_name], mode='lines', name=f"{scen}"))
    fig.update_layout(
        title=f'{buffer_name} — comparaison scénarios',
        xaxis_title='Temps (s)', yaxis_title='Pièces', template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )
    return fig.show()


# -----------------------------------------------------
# Diagnostics fusion (optionnel)
# -----------------------------------------------------

def plot_merge_weights(df: pd.DataFrame, title: Optional[str] = None, downsample: int = 1):
    if not {'wA','wB'}.issubset(df.columns):
        raise ValueError("Colonnes wA/wB absentes. Regénère les CSV avec diagnostics.")
    d = df.iloc[::max(1, int(downsample))]
    name = title or d.get('scenario', [''])[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['wA'], mode='lines', name='wA'))
    fig.add_trace(go.Scatter(x=d['time_s'], y=d['wB'], mode='lines', name='wB'))
    fig.update_layout(title=title or f'Poids de tirage (wA/wB) — {name}',
                      xaxis_title='Temps (s)', yaxis_title='Poids', template='plotly_white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0))
    fig.show()


# -----------------------------------------------------
# Démo rapide (exécutable)
# -----------------------------------------------------
if __name__ == '__main__':
    # Dossier data à côté de ce fichier par défaut
    script_dir = Path(__file__).resolve().parent
    data_dir = "data"
    data_dir_path = Path(data_dir).expanduser()
    data_dir_path.mkdir(parents=True, exist_ok=True)

    # Exemples de chemins
    p_nom = data_dir_path / 'flowline_nominal.csv'
    p_m2  = data_dir_path / 'flowline_m2_slow.csv'
    p_m3  = data_dir_path / 'flowline_m3_setup.csv'
    p_q1  = data_dir_path / 'flowline_m1_quality_drop.csv'

    # Charger un scénario et tracer
    if p_nom.exists():
        df_nom = load_csv(p_nom)
        plot_buffers(df_nom, title='Nominal')
        plot_flows(df_nom, title='Nominal')
        plot_counters(df_nom, title='Nominal')

    # Superposer s3_out sur plusieurs scénarios (si présents)
    present = [p for p in [p_nom, p_m2, p_m3, p_q1] if p.exists()]
    if len(present) >= 2:
        plot_throughput_overlay(present)
        plot_buffers_overlay(present, buffer_name='xC')
