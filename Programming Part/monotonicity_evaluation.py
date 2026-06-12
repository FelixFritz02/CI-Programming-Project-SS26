import numpy as np
import torch
import pandas as pd  


def evaluate_monotonicity_systematic(agent, env, compare_q_values: bool = False) -> tuple:
    """
    Testet Monotonie systematisch für jede Anfrage in der Instanz.

    Pro Anfrage i wird ein Statepaar konstruiert:
        s  = [t_i, C_max, ..., C_max, r_i, q_i]   ← volle Kapazität
        s' = [t_i, q_i_1, ..., q_i_K, r_i, q_i]   ← minimale Kapazität

    Geprüft wird aktionsweise: Q(s, a) ≥ Q(s', a) für alle gültigen a.

    Score = Anteil der (Anfrage, Aktion)-Paare die Monotonie erfüllen.
    """
    K        = env.K
    C_max    = max(env.C_k)
    instance = env._fixed_instance

    pairs_c = []
    pairs_t = []
    pairs_r = []

    for t_idx, request in enumerate(instance):
        t        = float(t_idx + 1)
        r        = float(request[0])
        q        = request[1:K + 1]
        cap_full = [float(C_max)] * K
        cap_min  = [float(C_max) - 1] * K

        # Paar 1: Monotonie in C_k
        s       = np.array([t] + cap_full + [r] + list(q), dtype=np.float32)
        s_prime = np.array([t] + cap_min  + [r] + list(q), dtype=np.float32)
        pairs_c.append((s, s_prime))

        # Paar 2: Monotonie in t (nur sinnvoll, wenn t+1 noch im gültigen Bereich liegt)
        if t_idx + 1 < len(instance):
            s_t       = np.array([t]       + cap_full + [r] + list(q), dtype=np.float32)
            s_t_prime = np.array([t + 1.0] + cap_full + [r] + list(q), dtype=np.float32)
            pairs_t.append((s_t, s_t_prime))

        # Paar 3: Monotonie in r (r vs. r+1, höherer Reward -> Q nicht kleiner)
        s_r       = np.array([t] + cap_full + [r]       + list(q), dtype=np.float32)
        s_r_prime = np.array([t] + cap_full + [r + 1.0] + list(q), dtype=np.float32)
        pairs_r.append((s_r, s_r_prime))

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
    score_t = _score(pairs_t, geq=False)  # Richtung anpassen je nach Semantik von t!
    score_r = _score(pairs_r, geq=False)  # r+1 -> Q sollte nicht kleiner sein als bei r (Richtung ggf. anpassen!)

    return score_c, score_t, score_r