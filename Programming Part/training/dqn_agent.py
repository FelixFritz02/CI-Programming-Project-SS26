import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # "Programming Part" auf sys.path (für env./models./training.-Importe)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models.combine_lattice_dqn import LatticeDQNNetwork
from models.combine_lattice_dqn_withaction import LatticeDQNNetworkWithAction
from training.monotonicity_evaluation import evaluate_monotonicity_systematic
from models.full_lattice import FullLatticeNetwork
from models.deep_lattice import DeepLatticeNetwork


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
    epsilon_decay_steps: Anzahl Umgebungsschritte (nicht Episoden!), über die
                   Epsilon exponentiell von epsilon_start auf epsilon_min
                   abklingt. Schrittbasiert statt episodenbasiert, damit die
                   Exploration nicht künstlich schnell abklingt, wenn
                   Episoden (z.B. durch frühen Kapazitäts-Abbruch) kurz sind.
    reject_bias  : Wahrscheinlichkeit, während Exploration Ablehnen (Aktion 0)
                   statt einer zufälligen Accept-Aktion zu wählen (0.0 = wie
                   bisher uniform über alle gültigen Aktionen).
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
        epsilon_min: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        reject_bias: float = 0.5,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        train_every: int = 4,
        warmup_steps: int = 500,
        tau: float = 0.005,
        max_grad_norm: float = 10.0,
        monotonicity_penalty: bool = False,
        mono_lambda: float = 0.1,
        mono_noise_scale: float = 3.0,
        constraint_every: int = 1,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.reject_bias = reject_bias
        self.batch_size = batch_size
        self.train_every = train_every
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.monotonicity_penalty = monotonicity_penalty
        self.mono_lambda = mono_lambda
        self.mono_noise_scale = mono_noise_scale
        # Monotonie-/Wertebereichs-Projektion (policy_net.apply_constraints) nur
        # alle `constraint_every` Trainingsschritte statt nach jedem — die
        # Constraint-Projektion ist bei Lattice-Netzen ein starker Laufzeitfaktor
        # (iterative Dykstra-Projektion pro Lattice/Calibrator), während die
        # Gewichte pro Schritt nur minimal driften. Am Ende von train() wird
        # einmal garantiert voll projiziert. <=1 = wie bisher jeder Schritt.
        self.constraint_every = max(1, int(constraint_every))

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
        elif isinstance(self.policy_net, LatticeDQNNetworkWithAction):
            self.optimizer = optim.Adam([
                {'params': self.policy_net.dqn.parameters(),           'lr': lr},
                {'params': self.policy_net.c_calibrators.parameters(), 'lr': lr * 5},
                {'params': self.policy_net.c_lattice.parameters(),     'lr': lr * 5},
                {'params': self.policy_net.action_calibrator.parameters(), 'lr': lr * 5},
            ])
        elif isinstance(self.policy_net, DeepLatticeNetwork):
            self.optimizer = optim.Adam([
                {'params': self.policy_net.cal_t.parameters(),        'lr': lr * 5},
                {'params': self.policy_net.cal_r.parameters(),        'lr': lr * 5},
                {'params': self.policy_net.cal_c.parameters(),        'lr': lr * 5},
                {'params': self.policy_net.cal_q.parameters(),        'lr': lr * 5},
                {'params': self.policy_net.lattices_A.parameters(),   'lr': lr * 5},
                {'params': self.policy_net.lattices_B.parameters(),   'lr': lr * 5},
                {'params': self.policy_net.lattices_C.parameters(),   'lr': lr * 5},
                {'params': self.policy_net.cal_between.parameters(),  'lr': lr * 5},
                {'params': self.policy_net.lattices_L2.parameters(),  'lr': lr * 5},
                {'params': self.policy_net.output_layer.parameters(), 'lr': lr},
            ])
        elif isinstance(self.policy_net, FullLatticeNetwork):
            self.optimizer = optim.Adam([
                {'params': self.policy_net.cal_t.parameters(),      'lr': lr * 5},
                {'params': self.policy_net.cal_r.parameters(),      'lr': lr * 5},
                {'params': self.policy_net.cal_c.parameters(),      'lr': lr * 5},
                {'params': self.policy_net.cal_q.parameters(),      'lr': lr * 5},
                {'params': self.policy_net.lattices_A.parameters(), 'lr': lr * 5},
                {'params': self.policy_net.lattices_B.parameters(),'lr': lr * 5},
                {'params': self.policy_net.lattices_C.parameters(), 'lr': lr * 5},
                {'params': self.policy_net.output_layer.parameters(), 'lr': lr},
            ])
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            
        self.loss_fn   = nn.MSELoss()

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Schrittzähler
        self._step = 0
        self._train_steps = 0   # Anzahl ausgeführter _train_step()-Aufrufe (für constraint_every)

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
    # Epsilon-Schedule (schrittbasiert)
    # ------------------------------------------------------------------

    def _decay_epsilon(self):
        """Exponentieller Abfall von epsilon_start auf epsilon_min über epsilon_decay_steps Umgebungsschritte."""
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(
            -self._step / self.epsilon_decay_steps
        )

    # ------------------------------------------------------------------
    # Aktionsauswahl (Epsilon-Greedy mit Action Masking)
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        valid_actions = self.env.get_valid_actions()

        # Exploration
        if random.random() < self.epsilon:
            # Ablehnen wird mit reject_bias bevorzugt (statt uniform über alle
            # gültigen Aktionen), damit zufällige Exploration die Kapazität
            # nicht sofort verbraucht und Episoden auch tiefer in die Instanz
            # kommen.
            accept_actions = [a for a in valid_actions if a != 0]
            if accept_actions and random.random() >= self.reject_bias:
                return random.choice(accept_actions)
            return 0

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
        if self.monotonicity_penalty:
            # ----------------------------------------------------------
            # Monotone Vergleichsstates erzeugen
            # C'_k >= C_k
            # ----------------------------------------------------------

            states_mono = states.clone()

            K = self.env.K

            # Zufällige positive Kapazitätserhöhung
            noise = torch.rand(
                (states.shape[0], K),
                dtype=torch.float32,
            ) * self.mono_noise_scale

            # Kapazitäten modifizieren
            states_mono[:, 1:K+1] += noise

            # Auf maximale Kapazität clippen
            max_cap = max(self.env.C_k)

            states_mono[:, 1:K+1] = torch.clamp(
                states_mono[:, 1:K+1],
                max=max_cap
            )
            # ----------------------------------------------------------
            # Monotonicity Penalty
            # Forderung:
            #   Q(states_mono, a) >= Q(states, a)
            # ----------------------------------------------------------

            q_orig = self.policy_net(states)
            q_mono = self.policy_net(states_mono)

            mono_violations = torch.relu(q_orig - q_mono)

            mono_loss = mono_violations.mean()
            
            loss = loss + self.mono_lambda * mono_loss

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping – verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Constraint-Projektion (Lattice-Netze) nur alle constraint_every
        # Trainingsschritte — siehe __init__. Der finale Voll-Projektionsschritt
        # passiert am Ende von train().
        self._train_steps += 1
        if self._train_steps % self.constraint_every == 0:
            self._apply_constraints()

        # Soft Update des Target-Netzes
        # θ_target = τ * θ_policy + (1 - τ) * θ_target
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

        return loss.item()

    def _apply_constraints(self):
        """Projiziert policy_net auf den zulässigen (monotonen, wertebereichs-
        konformen) Bereich zurück — no-op für Netze ohne apply_constraints()
        (z.B. Standard-DQN)."""
        if hasattr(self.policy_net, 'apply_constraints'):
            self.policy_net.apply_constraints()

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
        depth_history       : Liste von (episode, erreichte_tiefe, epsilon)-Tupeln;
                               erreichte_tiefe = Anzahl Umgebungsschritte, bevor
                               die Episode terminiert/truncated wurde
        """
        reward_history: list = []
        monotonicity_history: list = []
        depth_history: list = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            cumulated_reward = 0.0
            episode_depth = 0

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
                episode_depth += 1
                self._step += 1

                # 4. Epsilon reduzieren (schrittbasiert, nicht episodenbasiert)
                self._decay_epsilon()

                # 5. Trainieren (nur wenn genug Daten vorhanden)
                if len(self.buffer) >= self.warmup_steps and self._step % self.train_every == 0:
                    self._train_step()


            reward_history.append(cumulated_reward)
            depth_history.append((episode + 1, episode_depth, self.epsilon))

            if verbose and (episode + 1) % 50 == 0:
                avg = np.mean(reward_history[-50:])
                avg_depth = np.mean([d for _, d, _ in depth_history[-50:]])
                print(f"Episode {episode + 1:>4}/{num_episodes}  "
                      f"Reward: {cumulated_reward:>8.2f}  "
                      f"Avg(50): {avg:>8.2f}  "
                      f"Tiefe: {episode_depth:>4}  "
                      f"Avg-Tiefe(50): {avg_depth:>6.1f}  "
                      f"ε: {self.epsilon:.3f}")

            # Monotonie-Evaluation alle N Episoden
            if eval_monotonicity_every > 0 and (episode + 0) % eval_monotonicity_every == 0:
                mono_c, mono_t, mono_r, mono_q, mono_mixed = evaluate_monotonicity_systematic(self, self.env)
                monotonicity_history.append((episode + 1,  mono_c, mono_t, mono_r, mono_q, mono_mixed))
                if verbose:
                    print(f"  Systematic monotonicity (Episode {episode + 1}): "
                          f"C_k: {mono_c:.1%}, t: {mono_t:.1%}, r: {mono_r:.1%}, q: {mono_q:.1%}, mixed: {mono_mixed:.1%}")

        # Abschließende Voll-Projektion, damit das finale Netz garantiert die
        # Monotonie-/Wertebereichs-Constraints erfüllt (unabhängig davon, wo der
        # letzte Trainingsschritt relativ zu constraint_every lag).
        self._apply_constraints()

        return reward_history, monotonicity_history, depth_history

    # ------------------------------------------------------------------
    # Evaluationsschleife (rein exploitativ, ohne Training)
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 10_000, verbose: bool = True) -> tuple:
        """
        Evaluiert den (fertig trainierten) Agenten rein exploitativ.

        Epsilon wird auf 0 gesetzt (keine Exploration), das Netz läuft im
        Eval-Modus unter torch.no_grad(), und es finden keine Trainingsschritte,
        Buffer-Updates, Epsilon-Decays oder Soft-Updates statt. Die Gewichte
        von policy_net/target_net bleiben also unverändert (eingefroren).

        Der ursprüngliche epsilon-Wert wird am Ende wiederhergestellt, damit
        der Agent danach bei Bedarf weiterverwendet werden kann.

        Parameter
        ----------
        num_episodes : Anzahl Evaluierungsepisoden
        verbose       : Ausgabe nach je 500 Episoden

        Rückgabe
        --------
        reward_history: kumulierter Reward je Episode
        depth_history  : Liste von (episode, erreichte_tiefe)-Tupeln
        """
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        self.policy_net.eval()

        reward_history: list = []
        depth_history: list = []

        try:
            with torch.no_grad():
                for episode in range(num_episodes):
                    obs, _ = self.env.reset()
                    done = False
                    cumulated_reward = 0.0
                    episode_depth = 0

                    while not done:
                        action = self.select_action(obs)
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated

                        cumulated_reward += reward
                        episode_depth += 1

                    reward_history.append(cumulated_reward)
                    depth_history.append((episode + 1, episode_depth))

                    if verbose and (episode + 1) % 500 == 0:
                        avg = np.mean(reward_history[-500:])
                        avg_depth = np.mean([d for _, d in depth_history[-500:]])
                        print(f"[Eval] Episode {episode + 1:>5}/{num_episodes}  "
                              f"Reward: {cumulated_reward:>8.2f}  "
                              f"Avg(500): {avg:>8.2f}  "
                              f"Avg-Tiefe(500): {avg_depth:>6.1f}")
        finally:
            # Agent-Zustand wiederherstellen, unabhängig davon ob die Schleife
            # regulär durchlief oder z.B. per KeyboardInterrupt abgebrochen wurde.
            self.epsilon = original_epsilon
            self.policy_net.train()

        return reward_history, depth_history

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
    from env.gymnasium_env import DrauspEnv

    env = DrauspEnv(K=5, T_d=20, C_k=[20] * 5)
    agent = DQNAgent(env, lr=1e-3, gamma=0.9, epsilon_decay_steps=5_000)

    reward_history, monotonicity_history, depth_history = agent.train(num_episodes=500)

    print(f"\nBestes Ergebnis:       {max(reward_history):.2f}")
    print(f"Durchschnitt (letzte 50): {np.mean(reward_history[-50:]):.2f}")
