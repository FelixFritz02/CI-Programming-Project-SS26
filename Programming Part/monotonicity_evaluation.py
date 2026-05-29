import numpy as np
import torch
import pandas as pd  


def evaluate_monotonicity_systematic(agent, env, compare_q_values: bool = False) -> float:
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

    pairs = []

    for t_idx, request in enumerate(instance):
        t        = float(t_idx + 1)
        r        = float(request[0])
        q        = request[1:K + 1]
        cap_full = [float(C_max)] * K
        cap_min  = [float(q_k) for q_k in q]
        cap_min  = [float(C_max)-1] * K

        s       = np.array([t] + cap_full + [r] + list(q), dtype=np.float32)
        s_prime = np.array([t] + cap_min  + [r] + list(q), dtype=np.float32)
        pairs.append((s, s_prime))

    states       = torch.tensor(np.array([p[0] for p in pairs]), dtype=torch.float32)
    states_prime = torch.tensor(np.array([p[1] for p in pairs]), dtype=torch.float32)

    agent.policy_net.eval()
    with torch.no_grad():
        q_raw_s       = agent.policy_net(states)        # (T_d, A)
        q_raw_s_prime = agent.policy_net(states_prime)  # (T_d, A)

        q_masked_s       = agent._batch_mask_q_values(q_raw_s,       [p[0].tolist() for p in pairs])
        q_masked_s_prime = agent._batch_mask_q_values(q_raw_s_prime, [p[1].tolist() for p in pairs])
    agent.policy_net.train()

    # Nur Aktionen vergleichen die in BEIDEN States gültig sind
    valid_mask = (q_masked_s > -1e8) & (q_masked_s_prime > -1e8)  # (T_d, A)
    if compare_q_values:
        q_vergleich = pd.DataFrame({
            "q_s": q_masked_s.flatten().numpy(),
            "q_s_prime": q_masked_s_prime.flatten().numpy()})
        print(q_vergleich.head(20))
    # Monotonie pro (Anfrage, Aktion): Q(s,a) ≥ Q(s',a)
    correct = (q_masked_s >= q_masked_s_prime) & valid_mask        # (T_d, A)

    # Score: Anteil gültiger (Anfrage, Aktion)-Paare die Monotonie erfüllen
    score = correct[valid_mask].float().mean().item()

    return score