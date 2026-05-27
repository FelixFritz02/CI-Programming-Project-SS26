import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from lattice_dqn import LatticeDQNNetwork



# -----------------------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------------------

class ReplayBuffer:
    """Einfacher Experience-Replay-Buffer mit fester Maximalgröße."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards,                dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------

class DQNAgent:
    """
    DQN-Agent mit:
      - Action Masking  (ungültige Aktionen → Q = -inf)
      - Soft Update     (stabiler als harter Target-Network-Update)
      - Gradient Clipping (verhindert explodierende Gradienten)
      - Epsilon-Greedy  (Exploration vs. Exploitation)

    Parameter
    ----------
    env          : DrauspEnv  – Gymnasium-Umgebung
    lr           : Lernrate
    gamma        : Diskontierungsfaktor
    epsilon_start: Startwert für Epsilon (Exploration)
    epsilon_min  : Minimalwert für Epsilon
    epsilon_decay: Multiplikativer Abfall pro Episode
    batch_size   : Anzahl Samples pro Trainingsschritt
    buffer_size  : Maximale Größe des Replay Buffers
    train_every  : Trainingsschritt alle N Umgebungsschritte
    warmup_steps : Keine Trainingsschritte bevor Buffer diese Größe hat
    tau          : Soft-Update-Faktor (0 < tau << 1)
    max_grad_norm: Gradient Clipping Schwellwert
    """

    def __init__(
        self,
        env,
        QnetworkClass,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        train_every: int = 4,
        warmup_steps: int = 500,
        tau: float = 0.005,
        max_grad_norm: float = 10.0,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_every = train_every
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        # Dimensionen aus der Umgebung auslesen
        input_dim  = env.observation_space.shape[0]   # 2*K + 2
        output_dim = env.action_space.n               # K + 1

        # Netzwerke
        self.policy_net = QnetworkClass(input_dim, output_dim)
        self.target_net = QnetworkClass(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimierer & Loss
        if isinstance(self.policy_net, LatticeDQNNetwork):
            self.optimizer = optim.Adam([
                {'params': self.policy_net.dqn.parameters(),           'lr': lr},
                {'params': self.policy_net.c_calibrators.parameters(), 'lr': lr * 5},
                {'params': self.policy_net.c_lattice.parameters(),     'lr': lr * 5},
            ])
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            
        self.loss_fn   = nn.MSELoss()

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Schrittzähler
        self._step = 0

    # ------------------------------------------------------------------
    # Action Masking
    # ------------------------------------------------------------------

    def _mask_q_values(self, q_values: torch.Tensor, valid_actions: list) -> torch.Tensor:
        """Setzt Q-Werte ungültiger Aktionen auf -inf."""
        masked = torch.full_like(q_values, -1e9)
        masked[valid_actions] = q_values[valid_actions]
        return masked

    def _batch_mask_q_values(
        self,
        q_batch: torch.Tensor,
        states_list: list,
    ) -> torch.Tensor:
        """
        Vektorisiertes Action Masking für einen ganzen Batch.
        Erstellt eine Maske für alle Zustände gleichzeitig.
        """
        B, A = q_batch.shape
        mask = torch.full((B, A), -1e9)

        for i, state in enumerate(states_list):
            valid = self.env._get_valid_actions(state)
            mask[i, valid] = 0.0

        return q_batch + mask

    # ------------------------------------------------------------------
    # Aktionsauswahl (Epsilon-Greedy mit Action Masking)
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        valid_actions = self.env.get_valid_actions()

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).view(-1)

        masked_q = self._mask_q_values(q_values, valid_actions)
        return torch.argmax(masked_q).item()

    # ------------------------------------------------------------------
    # Trainingsschritt
    # ------------------------------------------------------------------

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Q(s, a) – aktuelles Netz
        q_values = self.policy_net(states).gather(1, actions)

        # Q(s', a') – Target-Netz mit Action Masking
        with torch.no_grad():
            q_next = self.target_net(next_states)

            # Vektorisiertes Action Masking über den gesamten Batch
            next_states_list = next_states.tolist()
            q_next = self._batch_mask_q_values(q_next, next_states_list)

            next_q_values = q_next.max(dim=1, keepdim=True)[0]

            # Bellman-Target
            # Wichtig: nur (1 - dones) wenn terminated, nicht truncated
            targets = rewards + self.gamma * (1 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping – verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

        self.optimizer.step()
        #neu hier das apply constraint für Lattice Networks
        if hasattr(self.policy_net, 'apply_constraints'):
            self.policy_net.apply_constraints()
        

        # Soft Update des Target-Netzes
        # θ_target = τ * θ_policy + (1 - τ) * θ_target
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

        return loss.item()

    # ------------------------------------------------------------------
    # Trainingsschleife
    # ------------------------------------------------------------------

    def train(
        self,
        num_episodes: int = 500,
        verbose: bool = True,
        eval_monotonicity_every: int = 50,
        monotonicity_pairs: int = 100,
    ) -> tuple:
        """
        Trainiert den Agenten für `num_episodes` Episoden.

        Alle `eval_monotonicity_every` Episoden wird `evaluate_monotonicity`
        aufgerufen und die Ratio geloggt.

        Parameter
        ----------
        num_episodes             : Anzahl Trainingsepisoden
        verbose                  : Ausgabe nach je 50 Episoden
        eval_monotonicity_every  : Intervall für Monotonie-Evaluation (0 = aus)
        monotonicity_pairs       : Anzahl Paare pro Evaluation

        Rückgabe
        --------
        reward_history      : kumulierter Reward je Episode
        monotonicity_history: Liste von (episode, ratio)-Tupeln
        """
        reward_history: list = []
        monotonicity_history: list = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            cumulated_reward = 0.0

            while not done:
                # 1. Aktion wählen
                action = self.select_action(obs)

                # 2. Umgebungsschritt
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 3. Erfahrung speichern
                # Nur terminated zählt für den Bellman-Update (nicht truncated)
                self.buffer.push(obs, action, reward, next_obs, float(terminated))

                obs = next_obs
                cumulated_reward += reward
                self._step += 1

                # 4. Trainieren (nur wenn genug Daten vorhanden)
                if len(self.buffer) >= self.warmup_steps and self._step % self.train_every == 0:
                    self._train_step()

            # Epsilon reduzieren
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            reward_history.append(cumulated_reward)

            if verbose and (episode + 1) % 50 == 0:
                avg = np.mean(reward_history[-50:])
                print(f"Episode {episode + 1:>4}/{num_episodes}  "
                      f"Reward: {cumulated_reward:>8.2f}  "
                      f"Avg(50): {avg:>8.2f}  "
                      f"ε: {self.epsilon:.3f}")

            # Monotonie-Evaluation alle N Episoden
            if eval_monotonicity_every > 0 and (episode + 1) % eval_monotonicity_every == 0:
                ratio = self.evaluate_monotonicity(num_pairs=monotonicity_pairs)
                monotonicity_history.append((episode + 1, ratio))
                if verbose:
                    print(f"  → Monotonie-Check (Episode {episode + 1:>4}): "
                          f"{ratio * 100:.1f}% korrekt ({monotonicity_pairs} Paare)")

        return reward_history, monotonicity_history

    # ------------------------------------------------------------------
    # Monotonie-Evaluation
    # ------------------------------------------------------------------

    def evaluate_monotonicity(self, num_pairs: int = 100) -> float:
        """
        Bewertet, wie monoton das aktuelle Q-Netz hinsichtlich der
        Restkapazitäten ist.

        Monotonie-Definition
        --------------------
        Für zwei Zustände s und s' mit identischem (t, r, q) gilt:
            s dominiert s'  ⟺  Ĉ_k(s) ≥ Ĉ_k(s') für alle k
                                UND Ĉ_k(s) > Ĉ_k(s') für mind. ein k

        Dann sollte das Q-Netz erfüllen:
            max_a Q(s, a) ≥ max_a Q(s', a)

        Mehr Restkapazität bedeutet mehr zukünftige Handlungsmöglichkeiten,
        also kann der erreichbare kumulierte Reward nie kleiner sein.

        Vorgehen
        ---------
        1. Sampele `num_pairs` Zustandspaare durch zufällige Episoden.
           Jedes Paar (s, s') teilt denselben Zeitschritt t sowie
           dieselbe Anfrage (r, q) und unterscheidet sich nur in den
           Restkapazitäten, wobei s alle Kapazitäten von s' dominiert.
        2. Prüfe für jedes Paar, ob max_a Q(s) ≥ max_a Q(s').
        3. Gib den Anteil korrekt geordneter Paare zurück (0.0 – 1.0).

        Parameter
        ----------
        num_pairs : Anzahl der zu sampelnden Zustandspaare.

        Rückgabe
        --------
        monotonicity_ratio : float ∈ [0, 1]
            Anteil der Paare, bei denen das Netz korrekt monoton ist.
        """
        K = self.env.K

        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        # Zustandspaare sammeln -------------------------------------------
        # Strategie: Pro Episode zwei zufällig verschiedene Kapazitätsvektoren
        # für denselben (t, r, q) konstruieren, sodass einer den anderen
        # komponentenweise dominiert.
        rng = np.random.default_rng()

        while len(pairs) < num_pairs:
            obs, _ = self.env.reset()
            done = False

            while not done and len(pairs) < num_pairs:
                # Aktueller Zeitschritt und aktuelle Anfrage aus obs
                t   = obs[0]
                r   = obs[K + 1]
                q   = obs[K + 2:]
                cap = obs[1 : K + 1]   # aktuelle Restkapazitäten

                # s' konstruieren: reduziere jeden Kapazitätseintrag um
                # einen zufälligen Betrag ∈ [0, cap_k], mindestens ein
                # Slot wird echt reduziert.
                reductions = rng.integers(cap.astype(int)/2, cap.astype(int) + 1, size=K)
                # Sicherstellen, dass Reduktion > 2
                if reductions.sum() > 2:
                    reductions[rng.integers(K)] = max(1, int(cap.max()))
                    reductions = np.minimum(reductions, cap.astype(int))

                cap_prime = cap - reductions.astype(np.float32)

                # Paare nur aufnehmen wenn cap_prime alle Werte ≥ 0 hat
                if np.all(cap_prime >= 0) and not np.array_equal(cap, cap_prime):
                    s       = np.array([t] + list(cap)       + [r] + list(q), dtype=np.float32)
                    s_prime = np.array([t] + list(cap_prime) + [r] + list(q), dtype=np.float32)
                    pairs.append((s, s_prime))

                # Zufällige gültige Aktion für nächsten Schritt
                valid = self.env.get_valid_actions()
                action = random.choice(valid)
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
        
        # Q-Werte berechnen -----------------------------------------------
        states       = torch.tensor(np.array([p[0] for p in pairs]), dtype=torch.float32)
        states_prime = torch.tensor(np.array([p[1] for p in pairs]), dtype=torch.float32)

        self.policy_net.eval()
        with torch.no_grad():
            q_raw_s       = self.policy_net(states)         # (num_pairs, A)
            q_raw_s_prime = self.policy_net(states_prime)   # (num_pairs, A)

            q_masked_s       = self._batch_mask_q_values(q_raw_s,       [p[0].tolist() for p in pairs])
            q_masked_s_prime = self._batch_mask_q_values(q_raw_s_prime, [p[1].tolist() for p in pairs])

            q_s       = q_masked_s.max(dim=1).values
            q_s_prime = q_masked_s_prime.max(dim=1).values
        self.policy_net.train()

        correct = (q_s >= q_s_prime).float().mean().item()

        return correct

    def save(self, path: str):
        """Speichert die Gewichte des Policy-Netzes."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Modell gespeichert: {path}")

    def load(self, path: str):
        """Lädt die Gewichte des Policy-Netzes."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Modell geladen: {path}")


# -----------------------------------------------------------------------
# Schnelltest
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium_env import DrauspEnv

    env = DrauspEnv(K=5, T_d=20, C_k=[20] * 5)
    agent = DQNAgent(env, lr=1e-3, gamma=0.9, epsilon_decay=0.995)

    reward_history, monotonicity_history = agent.train(num_episodes=500)

    print(f"\nBestes Ergebnis:       {max(reward_history):.2f}")
    print(f"Durchschnitt (letzte 50): {np.mean(reward_history[-50:]):.2f}")
