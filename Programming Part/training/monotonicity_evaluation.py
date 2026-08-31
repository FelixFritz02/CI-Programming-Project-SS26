import random

import numpy as np
import torch
import pandas as pd


def _sample_visited_states(env, num_rollouts: int = 5, num_states: int = None) -> list:
    """
    Rollt Episoden mit zufälligen gültigen Aktionen durch und sammelt jeden
    dabei tatsächlich besuchten Zustand (t, C, r, q). Liefert echte,
    ungleichmäßig abgebaute Kapazitäten statt synthetischer
    cap_full/cap_min/cap_half-Konstanten.

    num_states : ist ein Wert gesetzt, wird solange gerollt, bis mindestens so
                 viele Zustände gesammelt sind (danach auf num_states gekürzt) —
                 num_rollouts wird dann ignoriert. Sonst: genau num_rollouts
                 Episoden.
    """
    K = env.K
    visited = []
    rollout = 0
    while True:
        obs, _ = env.reset()
        done = False
        while not done:
            visited.append((float(obs[0]), list(obs[1:K + 1]), float(obs[K + 1]), list(obs[K + 2:])))
            valid_actions = env.get_valid_actions()
            action = random.choice(valid_actions)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        rollout += 1
        if num_states is not None:
            if len(visited) >= num_states:
                break
        elif rollout >= num_rollouts:
            break
    if num_states is not None and len(visited) > num_states:
        visited = visited[:num_states]
    return visited


def evaluate_monotonicity_systematic(agent, env, num_rollouts: int = 5, num_pairs: int = None) -> tuple:
    """
    Testet Monotonie an echten, per Zufalls-Rollout besuchten Zuständen
    (t, C, r, q) statt an synthetischen cap_full/cap_min/cap_half-Konstanten.

    Pro besuchtem Zustand wird je ein Vergleichspaar pro Achse gebaut:
        C_k  : Kapazität vs. Kapazität minus kleiner, nicht-uniformer Verringerung
        t    : gleiche Kapazität, ein Zeitschritt später
        r    : gleicher Zustand, doppelter Erlös
        q    : gleicher Zustand, halber Bedarf
        mixed: alle vier Achsen gleichzeitig verbessert

    Geprüft wird aktionsweise: Q(s, a) ≥ Q(s', a) für alle gültigen a.
    Score = Anteil der (Zustand, Aktion)-Paare, die Monotonie erfüllen.

    num_pairs : gewünschte Anzahl Vergleichspaare je Achse (= Anzahl besuchter
                Zustände). Ist der Wert gesetzt, wird so lange gerollt, bis so
                viele Zustände gesammelt sind; sonst genau num_rollouts Episoden.
    """
    K     = env.K
    C_max = max(env.C_k)

    pairs_c = []
    pairs_t = []
    pairs_r = []
    pairs_q = []
    pairs_mixed = []

    for t, C, r, q in _sample_visited_states(env, num_rollouts=num_rollouts, num_states=num_pairs):
        q_half = [qk / 2 for qk in q]

        # Paar 1: Monotonie in C_k (kleine, nicht-uniforme Verringerung pro Slot)
        down   = [random.uniform(0.05, 0.3) * C_max for _ in range(K)]
        C_less = [max(0.0, c - d) for c, d in zip(C, down)]
        pairs_c.append((
            np.array([t] + C + [r] + q, dtype=np.float32),
            np.array([t] + C_less + [r] + q, dtype=np.float32),
        ))

        # Paar 2: Monotonie in t (gleiche Kapazität, ein Zeitschritt später)
        pairs_t.append((
            np.array([t] + C + [r] + q, dtype=np.float32),
            np.array([t + 1.0] + C + [r] + q, dtype=np.float32),
        ))

        # Paar 3: Monotonie in r (r vs. 2r, höherer Erlös -> Q nicht kleiner)
        pairs_r.append((
            np.array([t] + C + [r] + q, dtype=np.float32),
            np.array([t] + C + [2 * r] + q, dtype=np.float32),
        ))

        # Paar 4: Monotonie in q (q vs. q/2, geringerer Bedarf -> Q nicht kleiner)
        pairs_q.append((
            np.array([t] + C + [r] + q, dtype=np.float32),
            np.array([t] + C + [r] + q_half, dtype=np.float32),
        ))

        # Paar 5: gemischte Monotonie - alle vier Achsen gleichzeitig verbessert
        if t > 2:
            up     = [random.uniform(0.05, 0.3) * C_max for _ in range(K)]
            C_more = [min(C_max, c + u) for c, u in zip(C, up)]
            pairs_mixed.append((
                np.array([t] + C + [r] + q, dtype=np.float32),
                np.array([t - 1.0] + C_more + [2 * r] + q_half, dtype=np.float32),
            ))

    
    def _score(pairs, geq=True):
        if len(pairs) == 0:
            return float("nan")

        states       = torch.tensor(np.array([p[0] for p in pairs]), dtype=torch.float32)
        states_prime = torch.tensor(np.array([p[1] for p in pairs]), dtype=torch.float32)

        agent.policy_net.eval()
        with torch.no_grad():
            q_raw_s       = agent.policy_net(states)
            q_raw_s_prime = agent.policy_net(states_prime)
            q_masked_s       = agent._batch_mask_q_values(q_raw_s,       [p[0].tolist() for p in pairs])
            q_masked_s_prime = agent._batch_mask_q_values(q_raw_s_prime, [p[1].tolist() for p in pairs])
        agent.policy_net.train()

        valid_mask = (q_masked_s > -1e8) & (q_masked_s_prime > -1e8)
        if geq:
            correct = (q_masked_s >= q_masked_s_prime) & valid_mask
        else:
            correct = (q_masked_s <= q_masked_s_prime) & valid_mask

        if valid_mask.sum().item() == 0:
            return float("nan")

        return correct[valid_mask].float().mean().item()

    score_c = _score(pairs_c, geq=True)
    score_t = _score(pairs_t, geq=True)  # Richtung anpassen je nach Semantik von t!
    score_r = _score(pairs_r, geq=False)  # r+1 -> Q sollte nicht kleiner sein als bei r (Richtung ggf. anpassen!)
    score_q = _score(pairs_q, geq=False)
    score_mixed = _score(pairs_mixed, geq=False)

    return score_c, score_t, score_r, score_q, score_mixed
