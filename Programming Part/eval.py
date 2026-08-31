"""
eval.py — Trainiert und evaluiert einen DQN-Agenten (Standard-DQN oder eine der
Lattice-Varianten) über alle (oder ausgewählte) Benchmark-Instanzen in `instances/`
und sammelt die Ergebnisse in einem pandas.DataFrame.

T_d (Anzahl Anfragen pro Episode) ist im DRAUSP nicht aus der Instanzdatei ablesbar
(siehe `env/instance_reader.py`) und wird daher über eine im Projekt abgelegte
Mapping-Datei `Programming Part/td_mapping.csv` bereitgestellt, die du selbst befüllst
(z.B. mit den Werten aus Tabelle 1/2 in `Literatur/drausp_lion18.pdf` für
lion18s/lion18w, frei wählbar für wendtris). Diese Datei ist Teil des Projekts
(eingecheckt), damit die Zuordnung für alle nachvollziehbar und wiederverwendbar ist.

Nutzung (CLI)
-------------
1. Einmalig die Mapping-Vorlage erzeugen (liegt danach unter td_mapping.csv; K,
   Pool-Größe, C_k etc. werden automatisch aus den Instanzdateien gelesen, nur die
   Spalte T_d muss von Hand befüllt werden):

       python eval.py template

   Die Spalte `paper_id_hint` zeigt dabei die zugehörige Paper-Notation an
   (z.B. Datei "SA01" -> Paper-Id "SA1"), um das Nachschlagen in den Tabellen
   zu erleichtern.

2. td_mapping.csv ausfüllen (Spalte T_d), dann Training + Evaluation starten
   (--td-mapping muss nicht mehr angegeben werden, Default ist td_mapping.csv):

       python eval.py run --model standard_dqn \
           --train-episodes 20000 --eval-episodes 10000 --out results.csv

Nutzung (aus Notebook/Code)
----------------------------
    from eval import evaluate_instances, discover_instances, load_td_mapping

    df = evaluate_instances(
        discover_instances(),
        load_td_mapping("td_mapping.csv"),
        model="deep_lattice",
        agent_kwargs=dict(gamma=1.0, reject_bias=0.75, epsilon_decay_steps=5000),
        num_train_episodes=20_000,
        num_eval_episodes=10_000,
        out_csv="results.csv",
    )
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # "Programming Part" auf sys.path

import argparse
import json
import random
import re
import time
import traceback
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from env.gymnasium_env import DrauspEnv
from env.instance_reader import get_instance_data
from training.dqn_agent import DQNAgent
from training.monotonicity_evaluation import evaluate_monotonicity_systematic
from models.standard_dqn import DQNNetwork
from models.combine_lattice_dqn import LatticeDQNNetwork
from models.combine_lattice_dqn_withaction import LatticeDQNNetworkWithAction
from models.full_lattice import FullLatticeNetwork
from models.deep_lattice import DeepLatticeNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "instances"
DEFAULT_TD_MAPPING = Path(__file__).resolve().parent / "td_mapping.csv"
DEFAULT_OUT_CSV = Path(__file__).resolve().parent / "eval_results.csv"

MODEL_CHOICES = ["standard_dqn"]


# =========================================================================
# EDITOR-KONFIGURATION
#
# Hier direkt anpassen und das Skript ohne Kommandozeilenargumente starten
# (z.B. mit dem "Run Python File"-Button oder F5). Wird nur verwendet, wenn
# eval.py OHNE Argumente aufgerufen wird — mit Argumenten (z.B. im Terminal
# "python eval.py run --model ...") greift stattdessen die CLI (siehe unten
# in main()), diese Werte werden dann ignoriert.
# =========================================================================
MODEL = "standard_dqn"          # einer aus MODEL_CHOICES
TRAIN_EPISODES = 20_000
EVAL_EPISODES = 10_000

SUBSETS = None                  # z.B. ["lion18s"], None = alle Unterordner (lion18s/lion18w/wendtris)
ONLY = None                     # z.B. ["SA01", "SA02"], None = alle Instanzen im/den Subset(s)
LIMIT = None                    # z.B. 3, um erst mal nur wenige Instanzen zum Testen zu laufen

DEFAULT_T_D = None              # Fallback-T_d für Instanzen ohne Eintrag in td_mapping.csv (z.B. wendtris)
AGENT_KWARGS = {
    "monotonicity_penalty": False,
    "reject_bias": 0.75,
    "gamma": 1,
    "epsilon_decay_steps": 5000,
}            
MODEL_KWARGS = {}               # Modell-kwargs, z.B. {"keypoints": 10} (nur für Lattice-Modelle relevant)

# Modellkomplexität matchen (z.B. standard_dqn gegen deep_lattice benchmarken):
# Nur wirksam wenn MODEL == "standard_dqn". Wird pro Instanz automatisch die
# Parameterzahl von MATCH_PARAMS_TO berechnet und dazu passende hidden_dims gesucht
# (überschreibt ein evtl. in MODEL_KWARGS gesetztes "hidden_dims"). None = aus.
MATCH_PARAMS_TO = "deep_lattice"          # z.B. "deep_lattice"
MATCH_PARAMS_TO_KWARGS = {}     # kwargs für das Zielmodell, z.B. {"keypoints": 8, "lattice_units": 4}
MATCH_DEPTH = 2                 # Anzahl Hidden-Layer für die standard_dqn-Suche
MATCH_REQUIRE_FUNNEL = True     # h1 >= h2 >= ... erzwingen (wie bisherige (32,16)-Architektur)

OUT_CSV = str(DEFAULT_OUT_CSV)  # liegt fest unter Programming Part/eval_results.csv
RESUME = False                  # True = an vorhandener OUT_CSV fortsetzen, fertige Instanzen überspringen
SEED = None
HISTORIES_DIR = None            # z.B. "histories" um reward/depth/mono-Verläufe je Instanz zu speichern
MODELS_DIR = None               # z.B. "trained_models" um die trainierten Gewichte je Instanz zu speichern

TRAIN_VERBOSE = False           # True = alle 50 Trainings-Episoden eine Zeile drucken (sehr viel Output)
EVAL_VERBOSE = False            # True = alle 500 Eval-Episoden eine Zeile drucken
EVAL_MONOTONICITY_EVERY = 100     # >0 aktiviert periodische Monotonie-Auswertung während des Trainings
MONOTONICITY_PAIRS = 1000
# >0: nach Training + Eval EINMAL die Monotonie des finalen Netzes mit so vielen
# Vergleichspaaren je Achse messen und als mono_*_final in die Ergebnis-CSV
# schreiben (unabhängig von EVAL_MONOTONICITY_EVERY). 0 = aus.
FINAL_MONOTONICITY_PAIRS = 10_000
# =========================================================================


# -----------------------------------------------------------------------
# Instanzen finden
# -----------------------------------------------------------------------

def discover_instances(
    instances_dir: Union[str, Path] = DEFAULT_INSTANCES_DIR,
    subsets: Optional[Sequence[str]] = None,
) -> list:
    """Sammelt alle *.txt-Instanzdateien unter `instances_dir`.

    `subsets` schränkt auf bestimmte Unterordner ein (z.B. ["lion18s"]);
    ohne Angabe werden alle vorhandenen Unterordner durchsucht.
    """
    instances_dir = Path(instances_dir)
    if subsets is None:
        subsets = sorted(p.name for p in instances_dir.iterdir() if p.is_dir())

    paths = []
    for subset in subsets:
        subset_dir = instances_dir / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Instanz-Unterordner nicht gefunden: {subset_dir}")
        paths.extend(sorted(subset_dir.glob("*.txt")))
    return paths


def _paper_id_hint(name: str) -> str:
    """Wandelt z.B. 'SA01' -> 'SA1', um die Paper-Tabellen-Notation zu erleichtern."""
    match = re.match(r"^([A-Za-z-]+?)0*(\d+)$", name)
    if match:
        prefix, num = match.groups()
        return f"{prefix}{num}"
    return name


# -----------------------------------------------------------------------
# T_d-Mapping (muss vom Nutzer befüllt werden, siehe Modul-Docstring)
# -----------------------------------------------------------------------

def generate_td_template(instance_paths: Sequence[Path], out_csv: Union[str, Path]) -> pd.DataFrame:
    """Erzeugt eine CSV-Vorlage mit einer leeren T_d-Spalte zum Ausfüllen.

    Alle anderen Spalten (K, pool_size, request_length, capacity_vector) werden
    direkt aus den Instanzdateien gelesen und sind daher verlässlich; nur T_d
    ist im DRAUSP frei wählbar bzw. muss aus der Paper-Tabelle übernommen werden.
    """
    rows = []
    for path in instance_paths:
        data = get_instance_data(path)
        rows.append({
            "subset": path.parent.name,
            "instance": path.stem,
            "paper_id_hint": _paper_id_hint(path.stem),
            "K": data.num_slots,
            "pool_size": data.pool_size,
            "request_length": data.request_length,
            "capacity_vector": " ".join(map(str, data.capacity_vector)),
            "T_d": "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Vorlage geschrieben: {out_csv} ({len(df)} Instanzen). "
          f"Bitte Spalte 'T_d' befüllen (siehe Tabelle 1/2 in drausp_lion18.pdf, "
          f"Zuordnung über 'paper_id_hint').")
    return df


def load_td_mapping(csv_path: Union[str, Path]) -> dict:
    """Lädt ein T_d-Mapping (Spalten 'instance', 'T_d') aus einer CSV-Datei.

    Zeilen mit leerem T_d werden übersprungen (Instanz muss dann `default_T_d`
    verwenden oder wird als Fehler markiert).
    """
    df = pd.read_csv(csv_path, dtype={"instance": str})
    mapping = {}
    for _, row in df.iterrows():
        val = row.get("T_d")
        if pd.isna(val) or str(val).strip() == "":
            continue
        mapping[str(row["instance"])] = int(val)
    return mapping


# -----------------------------------------------------------------------
# Modell-Auswahl
# -----------------------------------------------------------------------

def _coerce_ranges(kwargs: dict) -> dict:
    """JSON kennt keine Tupel — wandelt '*_range'-Listen in Tupel um."""
    return {
        k: (tuple(v) if k.endswith("_range") and isinstance(v, list) else v)
        for k, v in kwargs.items()
    }


def build_qnetwork_class(model: str, K: int, T_d: int, C_k: list, **model_kwargs):
    """Baut die `QnetworkClass`-Factory für `DQNAgent`, passend zur Instanz.

    Standard-DQN und die actionabhängige Lattice-Variante brauchen keine
    instanzspezifischen Wertebereiche. Full-/Deep-Lattice kalibrieren dagegen
    auf konkrete Wertebereiche (t, C_k, r, q_k), die hier automatisch aus der
    Instanz abgeleitet werden (wie in den Notebooks), sofern nicht explizit
    über model_kwargs überschrieben.
    """
    model_kwargs = _coerce_ranges(dict(model_kwargs))
    max_cap = float(max(C_k))

    if model == "standard_dqn":
        return lambda input_dim, output_dim: DQNNetwork(input_dim, output_dim, **model_kwargs)

    if model == "lattice_c":
        c_range = model_kwargs.pop("c_range", (0.0, max_cap))
        return lambda input_dim, output_dim: LatticeDQNNetwork(
            input_dim, output_dim, c_range=c_range, **model_kwargs
        )

    if model == "lattice_c_action":
        c_range = model_kwargs.pop("c_range", (0.0, max_cap))
        return lambda input_dim, output_dim: LatticeDQNNetworkWithAction(
            input_dim, output_dim, c_range=c_range, **model_kwargs
        )

    if model == "full_lattice":
        c_range = model_kwargs.pop("c_range", (0.0, max_cap))
        t_range = model_kwargs.pop("t_range", (1.0, float(T_d)))
        r_range = model_kwargs.pop("r_range", (0.0, 100.0))
        q_range = model_kwargs.pop("q_range", (0.0, max_cap))
        return lambda input_dim, output_dim: FullLatticeNetwork(
            input_dim, output_dim, c_range=c_range, t_range=t_range,
            r_range=r_range, q_range=q_range, **model_kwargs
        )

    if model == "deep_lattice":
        c_range = model_kwargs.pop("c_range", (0.0, max_cap))
        t_range = model_kwargs.pop("t_range", (1.0, float(T_d)))
        r_range = model_kwargs.pop("r_range", (0.0, 100.0))
        q_range = model_kwargs.pop("q_range", (0.0, max_cap))
        return lambda input_dim, output_dim: DeepLatticeNetwork(
            input_dim, output_dim, c_range=c_range, t_range=t_range,
            r_range=r_range, q_range=q_range, **model_kwargs
        )

    raise ValueError(f"Unbekanntes Modell '{model}'. Verfügbar: {MODEL_CHOICES}")


def count_parameters(model: str, instance_path: Union[str, Path], T_d: int, **model_kwargs) -> int:
    """Baut `model` für die gegebene Instanz auf (ohne zu trainieren) und zählt die
    lernbaren Parameter — zum Vergleich der Modellkomplexität zwischen z.B.
    standard_dqn (über model_kwargs={"hidden_dims": (...)} einstellbar) und
    deep_lattice/full_lattice (über keypoints/lattice_units/lattice_units2), bei
    identischem State-/Action-Space.

    input_dim/output_dim werden wie in DrauspEnv/DQNAgent aus der Instanz abgeleitet
    (input_dim = 2*K + 2, output_dim = K + 2 - request_length).
    """
    data = get_instance_data(Path(instance_path))
    input_dim = 2 * data.num_slots + 2
    output_dim = data.num_slots + 2 - data.request_length
    QnetworkClass = build_qnetwork_class(
        model, K=data.num_slots, T_d=T_d, C_k=data.capacity_vector, **model_kwargs
    )
    net = QnetworkClass(input_dim, output_dim)
    return sum(p.numel() for p in net.parameters())


def _mlp_param_count(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> int:
    """Geschlossene Parameterformel für ein DQNNetwork-MLP (Linear+ReLU je Hidden-Layer,
    kein Torch-Modellbau nötig, daher auch bei großem Suchraum schnell)."""
    prev_dim = input_dim
    total = 0
    for h in hidden_dims:
        total += (prev_dim + 1) * h  # Gewichte + Bias
        prev_dim = h
    total += (prev_dim + 1) * output_dim
    return total


def find_matching_hidden_dims(
    target_params: int,
    instance_path: Union[str, Path],
    depth: int = 2,
    require_funnel: bool = True,
    max_width: int = 300,
) -> tuple:
    """Sucht `hidden_dims` für ein `standard_dqn` (DQNNetwork) auf der gegebenen
    Instanz, dessen Parameterzahl möglichst nah an `target_params` liegt (z.B.
    `count_parameters("deep_lattice", instance_path, T_d)` derselben Instanz).

    Feste Tiefe `depth` (Anzahl Hidden-Layer, Default 2 — wie die bisherige
    (32, 16)-Architektur). Für depth<=2 wird exakt per Gitter über
    1..max_width je Layer gesucht; für depth>2 werden alle Hidden-Layer gleich
    breit gesucht (eine Suchdimension statt depth-vieler).

    require_funnel=True erzwingt nicht wachsende Breiten (h1 >= h2 >= ...), wie
    in der bisherigen Architektur (32, 16).

    Rückgabe: (hidden_dims, tatsächlich_erreichte_parameterzahl)
    """
    data = get_instance_data(Path(instance_path))
    input_dim = 2 * data.num_slots + 2
    output_dim = data.num_slots + 2 - data.request_length

    best_diff, best_hidden, best_n = None, None, None

    def consider(hidden_dims):
        nonlocal best_diff, best_hidden, best_n
        n = _mlp_param_count(input_dim, output_dim, hidden_dims)
        diff = abs(n - target_params)
        if best_diff is None or diff < best_diff:
            best_diff, best_hidden, best_n = diff, hidden_dims, n

    if depth == 1:
        for h in range(1, max_width + 1):
            consider((h,))
    elif depth == 2:
        for h1 in range(1, max_width + 1):
            h2_max = h1 if require_funnel else max_width
            for h2 in range(1, h2_max + 1):
                consider((h1, h2))
    else:
        # depth > 2: alle Hidden-Layer gleich breit (eindimensionale Suche)
        for h in range(1, max_width + 1):
            consider((h,) * depth)

    return best_hidden, best_n


# -----------------------------------------------------------------------
# Haupt-Evaluationsschleife
# -----------------------------------------------------------------------

def evaluate_instances(
    instance_paths: Sequence[Path],
    td_mapping: dict,
    model: str = "standard_dqn",
    model_kwargs: Optional[dict] = None,
    agent_kwargs: Optional[dict] = None,
    num_train_episodes: int = 20_000,
    num_eval_episodes: int = 10_000,
    default_T_d: Optional[int] = None,
    train_verbose: bool = False,
    eval_verbose: bool = False,
    eval_monotonicity_every: int = 0,
    monotonicity_pairs: int = 100,
    final_monotonicity_pairs: int = 0,
    seed: Optional[int] = None,
    out_csv: Optional[Union[str, Path]] = None,
    resume: bool = False,
    histories_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    match_params_to: Optional[str] = None,
    match_params_to_kwargs: Optional[dict] = None,
    match_depth: int = 2,
    match_require_funnel: bool = True,
) -> pd.DataFrame:
    """Trainiert+evaluiert `model` auf jeder Instanz in `instance_paths`.

    Für jede Instanz wird eine frische Umgebung/ein frischer Agent erzeugt
    (kein Transfer-Learning zwischen Instanzen), `num_train_episodes` trainiert
    und anschließend rein exploitativ über `num_eval_episodes` evaluiert.

    T_d wird pro Instanz aus `td_mapping` (Schlüssel = Dateiname ohne .txt)
    gelesen; fehlt ein Eintrag, greift `default_T_d` (falls gesetzt), sonst
    wird die Instanz mit einer Fehlermeldung übersprungen.

    Nach jeder Instanz wird das DataFrame (bei gesetztem `out_csv`) neu
    geschrieben, damit bei einem Abbruch (Laufzeit, KeyboardInterrupt, Fehler)
    kein bereits berechnetes Ergebnis verloren geht. Mit `resume=True` werden
    beim erneuten Start bereits in `out_csv` vorhandene Instanzen übersprungen.

    Monotonie:
    `eval_monotonicity_every > 0` misst die Monotonie periodisch WÄHREND des
    Trainings (billig, wenige Rollouts) und speichert den Verlauf in die
    histories. `final_monotonicity_pairs > 0` misst die Monotonie zusätzlich
    EINMAL nach Training + Eval am finalen Netz mit so vielen Vergleichspaaren je
    Achse (z.B. 10_000) und schreibt sie als `mono_c_final … mono_mixed_final`
    (+ `mono_source="final_net"`, `mono_final_pairs`) in jede Ergebniszeile. Ohne
    `final_monotonicity_pairs` fällt die CSV auf den letzten Trainings-Snapshot
    zurück (`mono_source="train_last"`), sofern `eval_monotonicity_every > 0`.

    Modellkomplexität matchen (z.B. standard_dqn gegen deep_lattice benchmarken):
    Ist `model == "standard_dqn"` und `match_params_to` gesetzt (Name eines anderen
    Modells aus MODEL_CHOICES), wird PRO INSTANZ automatisch die Parameterzahl von
    `match_params_to` (gebaut mit `match_params_to_kwargs`) berechnet und dazu per
    `find_matching_hidden_dims` ein passendes `hidden_dims` gesucht, BEVOR die
    Instanz trainiert wird — überschreibt ein evtl. in `model_kwargs` gesetztes
    `hidden_dims` für diese Instanz. Die tatsächliche Parameterzahl des trainierten
    Modells (`n_params`) sowie ggf. Zielmodell/Zielparameterzahl werden in jeder
    Ergebniszeile mitgeloggt, damit die Modellkomplexität im Bericht belegbar ist.
    """
    model_kwargs = model_kwargs or {}
    agent_kwargs = agent_kwargs or {}
    match_params_to_kwargs = match_params_to_kwargs or {}

    rows = []
    done_instances = set()
    if resume and out_csv is not None and Path(out_csv).exists():
        existing = pd.read_csv(out_csv, dtype={"instance": str})
        rows = existing.to_dict("records")
        done_instances = set(existing["instance"])
        print(f"Resume: {len(done_instances)} bereits vorhandene Instanzen werden übersprungen.")

    pbar = tqdm(instance_paths, desc=f"Instanzen ({model})")
    try:
        for path in pbar:
            name = path.stem
            subset = path.parent.name
            pbar.set_postfix(instance=name)

            if resume and name in done_instances:
                continue

            T_d = td_mapping.get(name, default_T_d)
            if T_d is None:
                print(f"[eval.py] Überspringe '{name}': kein T_d in td_mapping und kein default_T_d gesetzt.")
                rows.append({
                    "model": model, "subset": subset, "instance": name,
                    "T_d": None, "error": "no_T_d",
                })
                _flush(rows, out_csv)
                continue

            try:
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                data = get_instance_data(path)
                tqdm.write(f"-> {name} ({subset}, K={data.num_slots}, T_d={T_d}, |N|={data.pool_size})")

                instance_model_kwargs = dict(model_kwargs)
                match_info = {}
                if match_params_to is not None and model == "standard_dqn":
                    target_params = count_parameters(
                        match_params_to, path, T_d, **match_params_to_kwargs
                    )
                    hidden_dims, achieved_params = find_matching_hidden_dims(
                        target_params, path, depth=match_depth, require_funnel=match_require_funnel,
                    )
                    instance_model_kwargs["hidden_dims"] = hidden_dims
                    match_info = {
                        "match_target_model": match_params_to,
                        "match_target_params": target_params,
                        "match_hidden_dims": str(hidden_dims),
                    }
                    tqdm.write(f"   match_params_to={match_params_to}: target={target_params} "
                               f"-> hidden_dims={hidden_dims} ({achieved_params} params)")

                env = DrauspEnv(
                    K=data.num_slots,
                    T_d=T_d,
                    C_k=data.capacity_vector,
                    instance=data.instance,
                    fixed_request_length=data.request_length,
                )
                QnetworkClass = build_qnetwork_class(
                    model, K=data.num_slots, T_d=T_d, C_k=data.capacity_vector, **instance_model_kwargs
                )
                agent = DQNAgent(env, QnetworkClass=QnetworkClass, **agent_kwargs)
                n_params = sum(p.numel() for p in agent.policy_net.parameters())

                t0 = time.time()
                reward_hist, mono_hist, depth_hist = agent.train(
                    num_episodes=num_train_episodes,
                    verbose=train_verbose,
                    eval_monotonicity_every=eval_monotonicity_every,
                    monotonicity_pairs=monotonicity_pairs,
                )
                train_seconds = time.time() - t0

                t1 = time.time()
                eval_rewards, eval_depths = agent.evaluate(
                    num_episodes=num_eval_episodes, verbose=eval_verbose
                )
                eval_seconds = time.time() - t1

                # Einmalige Monotonie-Messung des FINALEN Netzes (nach Training + Eval),
                # mit final_monotonicity_pairs Vergleichspaaren je Achse.
                mono_final = None
                mono_final_seconds = None
                if final_monotonicity_pairs and final_monotonicity_pairs > 0:
                    t2 = time.time()
                    mono_final = evaluate_monotonicity_systematic(
                        agent, env, num_pairs=final_monotonicity_pairs
                    )
                    mono_final_seconds = time.time() - t2
                    mc, mt, mr, mq, mm = mono_final
                    tqdm.write(f"   final monotonicity ({final_monotonicity_pairs} Paare/Achse): "
                               f"C_k={mc:.1%}, t={mt:.1%}, r={mr:.1%}, q={mq:.1%}, mixed={mm:.1%}")

                row = _build_result_row(
                    model=model, subset=subset, name=name, data=data, T_d=T_d,
                    num_train_episodes=num_train_episodes, num_eval_episodes=num_eval_episodes,
                    train_seconds=train_seconds, eval_seconds=eval_seconds,
                    reward_hist=reward_hist, mono_hist=mono_hist,
                    eval_rewards=eval_rewards, eval_depths=eval_depths,
                    final_train_epsilon=agent.epsilon, n_params=n_params,
                    mono_final=mono_final, mono_final_pairs=final_monotonicity_pairs,
                    mono_final_seconds=mono_final_seconds,
                )
                row.update(match_info)

                if histories_dir is not None:
                    _save_histories(
                        histories_dir, model, name,
                        reward_hist, depth_hist, mono_hist, eval_rewards, eval_depths,
                    )
                if models_dir is not None:
                    md = Path(models_dir)
                    md.mkdir(parents=True, exist_ok=True)
                    agent.save(str(md / f"{model}__{name}.pt"))

            except Exception as exc:
                print(f"[eval.py] Instanz '{name}' fehlgeschlagen: {exc}")
                traceback.print_exc()
                row = {
                    "model": model, "subset": subset, "instance": name, "T_d": T_d,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            rows.append(row)
            _flush(rows, out_csv)
    except KeyboardInterrupt:
        print("\n[eval.py] Abgebrochen (KeyboardInterrupt) — bisherige Ergebnisse werden zurückgegeben.")

    return pd.DataFrame(rows)


def _build_result_row(
    model, subset, name, data, T_d,
    num_train_episodes, num_eval_episodes,
    train_seconds, eval_seconds,
    reward_hist, mono_hist, eval_rewards, eval_depths, final_train_epsilon, n_params,
    mono_final=None, mono_final_pairs=0, mono_final_seconds=None,
) -> dict:
    eval_r = np.array(eval_rewards, dtype=float)
    eval_d = np.array([d for _, d in eval_depths], dtype=float)
    train_tail = np.array(reward_hist[-100:], dtype=float) if reward_hist else np.array([])

    row = {
        "model": model,
        "subset": subset,
        "instance": name,
        "K": data.num_slots,
        "pool_size": data.pool_size,
        "request_length": data.request_length,
        "T_d": T_d,
        "capacity_vector": " ".join(map(str, data.capacity_vector)),
        "train_episodes": num_train_episodes,
        "eval_episodes": num_eval_episodes,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "train_reward_last100_mean": float(train_tail.mean()) if len(train_tail) else float("nan"),
        "eval_reward_mean": float(eval_r.mean()),
        "eval_reward_std": float(eval_r.std(ddof=1)) if len(eval_r) > 1 else 0.0,
        "eval_reward_median": float(np.median(eval_r)),
        "eval_reward_min": float(eval_r.min()),
        "eval_reward_max": float(eval_r.max()),
        "eval_depth_mean": float(eval_d.mean()) if len(eval_d) else float("nan"),
        "n_params": n_params,
        # Epsilon-Stand am ENDE DES TRAININGS (nicht der Eval-Phase — dort ist Epsilon
        # in DQNAgent.evaluate() immer fest auf 0.0 gesetzt, siehe training/dqn_agent.py).
        # Dient nur als Diagnose, ob Epsilon bis Trainingsende auf epsilon_min abgeklungen ist.
        "final_train_epsilon": final_train_epsilon,
        "error": None,
    }
    # Monotonie-Spalten: bevorzugt die dedizierte Messung des finalen Netzes
    # (mono_final aus evaluate_monotonicity_systematic mit vielen Paaren), sonst
    # als Fallback der letzte periodische Trainings-Snapshot (mono_hist[-1]).
    if mono_final is not None:
        mono_c, mono_t, mono_r, mono_q, mono_mixed = mono_final
        row.update({
            "mono_c_final": mono_c, "mono_t_final": mono_t, "mono_r_final": mono_r,
            "mono_q_final": mono_q, "mono_mixed_final": mono_mixed,
            "mono_source": "final_net",
            "mono_final_pairs": mono_final_pairs,
            "mono_final_seconds": mono_final_seconds,
        })
    elif mono_hist:
        _, mono_c, mono_t, mono_r, mono_q, mono_mixed = mono_hist[-1]
        row.update({
            "mono_c_final": mono_c, "mono_t_final": mono_t, "mono_r_final": mono_r,
            "mono_q_final": mono_q, "mono_mixed_final": mono_mixed,
            "mono_source": "train_last",
        })
    return row


def _save_histories(histories_dir, model, name, reward_hist, depth_hist, mono_hist, eval_rewards, eval_depths):
    hd = Path(histories_dir)
    hd.mkdir(parents=True, exist_ok=True)
    np.savez(
        hd / f"{model}__{name}.npz",
        reward_history=np.array(reward_hist, dtype=float),
        depth_history=np.array(depth_hist, dtype=object),
        monotonicity_history=np.array(mono_hist, dtype=object),
        eval_rewards=np.array(eval_rewards, dtype=float),
        eval_depths=np.array(eval_depths, dtype=object),
    )


def _flush(rows: list, out_csv: Optional[Union[str, Path]]):
    if out_csv is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _add_agent_arguments(parser: argparse.ArgumentParser):
    """Spiegelt die Hyperparameter von DQNAgent (siehe training/dqn_agent.py)."""
    g = parser.add_argument_group("DQNAgent-Hyperparameter")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--gamma", type=float, default=0.9)
    g.add_argument("--epsilon-start", type=float, default=1.0)
    g.add_argument("--epsilon-min", type=float, default=0.1)
    g.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    g.add_argument("--reject-bias", type=float, default=0.5)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--buffer-size", type=int, default=50_000)
    g.add_argument("--train-every", type=int, default=4)
    g.add_argument("--warmup-steps", type=int, default=500)
    g.add_argument("--tau", type=float, default=0.005)
    g.add_argument("--max-grad-norm", type=float, default=10.0)
    g.add_argument("--monotonicity-penalty", action="store_true")
    g.add_argument("--mono-lambda", type=float, default=0.1)
    g.add_argument("--mono-noise-scale", type=float, default=3.0)
    g.add_argument("--constraint-every", type=int, default=10,
                    help="Constraint-Projektion (Lattice-Netze) nur alle N Trainingsschritte "
                         "(1 = nach jedem Schritt); Voll-Projektion am Trainingsende immer")
    g.add_argument("--agent-kwargs-json", type=str, default=None,
                    help="Zusätzliche/überschreibende DQNAgent-kwargs als JSON, z.B. "
                         '\'{"buffer_size": 100000}\'')


def _agent_kwargs_from_args(args) -> dict:
    kwargs = dict(
        lr=args.lr, gamma=args.gamma, epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min, epsilon_decay_steps=args.epsilon_decay_steps,
        reject_bias=args.reject_bias, batch_size=args.batch_size,
        buffer_size=args.buffer_size, train_every=args.train_every,
        warmup_steps=args.warmup_steps, tau=args.tau, max_grad_norm=args.max_grad_norm,
        monotonicity_penalty=args.monotonicity_penalty, mono_lambda=args.mono_lambda,
        mono_noise_scale=args.mono_noise_scale, constraint_every=args.constraint_every,
    )
    if args.agent_kwargs_json:
        kwargs.update(json.loads(args.agent_kwargs_json))
    return kwargs


def _run_from_editor_config():
    """Nutzt den EDITOR-KONFIGURATION-Block oben in der Datei (kein CLI-Aufruf nötig)."""
    paths = discover_instances(DEFAULT_INSTANCES_DIR, SUBSETS)
    if ONLY:
        only = set(ONLY)
        paths = [p for p in paths if p.stem in only]
    if LIMIT:
        paths = paths[:LIMIT]
    if not paths:
        raise SystemExit("Keine Instanzen gefunden (SUBSETS/ONLY/LIMIT im Konfig-Block prüfen).")

    td_mapping = load_td_mapping(DEFAULT_TD_MAPPING)

    df = evaluate_instances(
        paths, td_mapping,
        model=MODEL,
        model_kwargs=MODEL_KWARGS,
        agent_kwargs=AGENT_KWARGS,
        num_train_episodes=TRAIN_EPISODES,
        num_eval_episodes=EVAL_EPISODES,
        default_T_d=DEFAULT_T_D,
        train_verbose=TRAIN_VERBOSE,
        eval_verbose=EVAL_VERBOSE,
        eval_monotonicity_every=EVAL_MONOTONICITY_EVERY,
        monotonicity_pairs=MONOTONICITY_PAIRS,
        final_monotonicity_pairs=FINAL_MONOTONICITY_PAIRS,
        seed=SEED,
        out_csv=OUT_CSV,
        resume=RESUME,
        histories_dir=HISTORIES_DIR,
        models_dir=MODELS_DIR,
        match_params_to=MATCH_PARAMS_TO,
        match_params_to_kwargs=MATCH_PARAMS_TO_KWARGS,
        match_depth=MATCH_DEPTH,
        match_require_funnel=MATCH_REQUIRE_FUNNEL,
    )
    print(df)
    print(f"\nErgebnisse geschrieben: {OUT_CSV}")


def main():
    if len(sys.argv) == 1:
        _run_from_editor_config()
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_template = sub.add_parser("template", help="T_d-Mapping-Vorlage aus instances/ erzeugen")
    p_template.add_argument("--instances-dir", type=str, default=str(DEFAULT_INSTANCES_DIR))
    p_template.add_argument("--subsets", nargs="+", default=None)
    p_template.add_argument("--out", type=str, default=str(DEFAULT_TD_MAPPING))

    p_run = sub.add_parser("run", help="Training + Evaluation über alle Instanzen ausführen")
    p_run.add_argument("--instances-dir", type=str, default=str(DEFAULT_INSTANCES_DIR))
    p_run.add_argument("--subsets", nargs="+", default=None,
                        help="z.B. --subsets lion18s lion18w (Default: alle Unterordner)")
    p_run.add_argument("--only", nargs="+", default=None,
                        help="Nur diese Instanznamen (Dateiname ohne .txt) verwenden")
    p_run.add_argument("--limit", type=int, default=None, help="Nur die ersten N Instanzen (zum Testen)")
    p_run.add_argument("--td-mapping", type=str, default=str(DEFAULT_TD_MAPPING),
                        help="CSV aus 'template' (Spalte T_d befüllt); liegt standardmäßig unter "
                             "Programming Part/td_mapping.csv")
    p_run.add_argument("--default-td", type=int, default=None,
                        help="T_d für Instanzen ohne Eintrag in --td-mapping (z.B. für wendtris)")
    p_run.add_argument("--model", choices=MODEL_CHOICES, default="standard_dqn")
    p_run.add_argument("--model-kwargs-json", type=str, default=None,
                        help='Modell-spezifische kwargs als JSON, z.B. \'{"keypoints": 10}\'')
    p_run.add_argument("--train-episodes", type=int, default=20_000)
    p_run.add_argument("--eval-episodes", type=int, default=10_000)
    p_run.add_argument("--eval-monotonicity-every", type=int, default=0)
    p_run.add_argument("--monotonicity-pairs", type=int, default=100)
    p_run.add_argument("--final-monotonicity-pairs", type=int, default=0,
                        help="Nach Training + Eval EINMAL die Monotonie des finalen Netzes mit "
                             "so vielen Vergleichspaaren je Achse messen und als mono_*_final in "
                             "die CSV schreiben (0 = aus)")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV))
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--histories-dir", type=str, default=None)
    p_run.add_argument("--models-dir", type=str, default=None)
    p_run.add_argument("--train-verbose", action="store_true")
    p_run.add_argument("--eval-verbose", action="store_true")
    p_run.add_argument("--match-params-to", type=str, default=None, choices=MODEL_CHOICES,
                        help="Nur bei --model standard_dqn: pro Instanz automatisch hidden_dims "
                             "suchen, die die Parameterzahl dieses Modells matchen")
    p_run.add_argument("--match-params-to-kwargs-json", type=str, default=None)
    p_run.add_argument("--match-depth", type=int, default=2)
    p_run.add_argument("--match-no-funnel", action="store_true",
                        help="h1 >= h2 >= ... NICHT erzwingen bei --match-params-to")
    _add_agent_arguments(p_run)

    args = parser.parse_args()

    if args.command == "template":
        paths = discover_instances(args.instances_dir, args.subsets)
        generate_td_template(paths, args.out)
        return

    if args.command == "run":
        paths = discover_instances(args.instances_dir, args.subsets)
        if args.only:
            only = set(args.only)
            paths = [p for p in paths if p.stem in only]
        if args.limit:
            paths = paths[: args.limit]
        if not paths:
            raise SystemExit("Keine Instanzen gefunden (Filter/Pfade prüfen).")

        td_mapping = load_td_mapping(args.td_mapping)
        model_kwargs = json.loads(args.model_kwargs_json) if args.model_kwargs_json else {}
        agent_kwargs = _agent_kwargs_from_args(args)
        match_params_to_kwargs = (
            json.loads(args.match_params_to_kwargs_json) if args.match_params_to_kwargs_json else {}
        )

        df = evaluate_instances(
            paths, td_mapping,
            model=args.model,
            model_kwargs=model_kwargs,
            agent_kwargs=agent_kwargs,
            num_train_episodes=args.train_episodes,
            num_eval_episodes=args.eval_episodes,
            default_T_d=args.default_td,
            train_verbose=args.train_verbose,
            eval_verbose=args.eval_verbose,
            eval_monotonicity_every=args.eval_monotonicity_every,
            monotonicity_pairs=args.monotonicity_pairs,
            final_monotonicity_pairs=args.final_monotonicity_pairs,
            seed=args.seed,
            out_csv=args.out,
            resume=args.resume,
            histories_dir=args.histories_dir,
            models_dir=args.models_dir,
            match_params_to=args.match_params_to,
            match_params_to_kwargs=match_params_to_kwargs,
            match_depth=args.match_depth,
            match_require_funnel=not args.match_no_funnel,
        )
        print(df)
        print(f"\nErgebnisse geschrieben: {args.out}")


if __name__ == "__main__":
    main()
