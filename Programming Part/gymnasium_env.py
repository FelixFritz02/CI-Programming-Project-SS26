import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

# Importiere deine bestehenden Module
from instance_generator import instance_generator
from instance_reader import get_instance_data


class DrauspEnv(gym.Env):
    """
    Gymnasium-Umgebung für das DRAUSP-Problem
    (Dynamic Resource Allocation Under Stochastic Prices).

    Zustandsraum:
        S = (t, Ĉ_1, ..., Ĉ_K, r, q_1, ..., q_K)
        - t:   aktueller Zeitschritt (1-indiziert)
        - Ĉ_k: verbleibende Kapazität in Slot k
        - r:   Erlös der aktuellen Anfrage
        - q_k: benötigte Kapazität der aktuellen Anfrage in Slot k

    Aktionsraum:
        {0, 1, ..., K}
        - 0:   Anfrage ablehnen
        - k>0: Anfrage ab Slot k einplanen

    Reward:
        - Ablehnen: 0
        - Akzeptieren (gültig): r (Erlös der Anfrage)
        - Ungültige Aktion:    -100 (Kapazität überschritten oder Slot ungültig)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        K: int = 5,
        T_d: int = 20,
        C_k: Optional[list] = None,
        lam: float = 1.0,
        instance=None,          # Feste Instanz (z.B. aus instance_reader)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.K = K
        self.T_d = T_d
        self.lam = lam
        self.render_mode = render_mode

        # Kapazitätsvektor
        self.C_k = C_k if C_k is not None else [20] * K

        # Optionale feste Instanz (Benchmark-Daten)
        self._fixed_instance = instance

        # -------------------------------------------------------------------
        # Aktionsraum: 0 = ablehnen, 1..K = Startslot
        # -------------------------------------------------------------------
        self.action_space = spaces.Discrete(K + 1)

        # -------------------------------------------------------------------
        # Beobachtungsraum:
        #   [t, Ĉ_1..Ĉ_K, r, q_1..q_K]  → Länge 2*K + 2
        #
        # Obergrenzen:
        #   t        ∈ [1, T_d]
        #   Ĉ_k      ∈ [0, max(C_k)]
        #   r        ∈ [0, ∞)   → wir nutzen einen großen Wert als Proxy
        #   q_k      ∈ [0, max(C_k)]
        # -------------------------------------------------------------------
        max_cap = max(self.C_k)
        low  = np.zeros(2 * K + 2, dtype=np.float32)
        high = np.array(
            [T_d]           # t
            + [max_cap] * K # Ĉ_k
            + [1e6]         # r  (Erlös unbeschränkt)
            + [max_cap] * K,# q_k
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Interner Zustand
        self._state = None
        self._instance = None

    # ------------------------------------------------------------------
    # Hilfsfunktionen (analog zu start.ipynb)
    # ------------------------------------------------------------------

    def _length_of_request(self, state):
        """Gibt die Länge (Anzahl belegter Slots) der aktuellen Anfrage zurück."""
        q = state[self.K + 2:]          # q-Vektor beginnt bei Index K+2
        for i, val in enumerate(q):
            if val == 0:
                return i
        return len(q)

    def _get_valid_actions(self, state):
        """Gibt eine Liste aller gültigen Aktionen zurück."""
        K = self.K
        valid = [0]                     # Ablehnen ist immer gültig
        len_q = self._length_of_request(state)
        for a in range(1, K + 2 - len_q):
            feasible = True
            for i in range(len_q):
                if state[a + i] < state[K + 2 + i]:
                    feasible = False
                    break
            if feasible:
                valid.append(a)
        return valid

    def _build_state(self, t, capacities, request):
        """Baut den Zustandsvektor auf."""
        return [t] + list(capacities) + [request[0]] + list(request[1:self.K + 1])

    # ------------------------------------------------------------------
    # Gymnasium-Interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None, show_instance=False):
        super().reset(seed=seed)

        # Neue oder feste Instanz laden
        if self._fixed_instance is not None:
            self._instance = self._fixed_instance
        else:
            self._instance = instance_generator(self.T_d, self.K, self.lam)
        
        if show_instance:
            print("Geladene Instanz:")
            for i, req in enumerate(self._instance):
                print(f"  Anfrage {i + 1}: {req}")

        self._capacities = list(self.C_k)  # verbleibende Kapazitäten
        self._t = 1                        # aktueller Zeitschritt

        first_request = self._instance[0]
        self._state = self._build_state(self._t, self._capacities, first_request)

        obs = np.array(self._state, dtype=np.float32)
        info = {"valid_actions": self._get_valid_actions(self._state)}
        return obs, info

    def step(self, action):
        state = self._state
        K = self.K
        len_q = self._length_of_request(state)

        terminated = False
        truncated  = False

        # --- Ungültige Aktion: Slot zu weit rechts ---
        if action > 0 and (len_q + action - 1) > K:
            reward = -100.0
            terminated = True
            obs = np.array(state, dtype=np.float32)
            info = {"valid_actions": []}
            return obs, reward, terminated, truncated, info

        # --- Aktion ausführen ---
        if action == 0:
            # Ablehnen
            reward = 0.0
            new_caps = list(state[1:K + 1])
        else:
            # Akzeptieren ab Slot `action`
            reward = float(state[K + 1])   # Erlös der aktuellen Anfrage
            new_caps = list(state[1:K + 1])
            for i in range(len_q):
                new_caps[action - 1 + i] -= state[K + 2 + i]

        # --- Kapazitätsverletzung prüfen ---
        if any(c < 0 for c in new_caps):
            reward = -100.0
            terminated = True
            # Zustand trotzdem aktualisieren (für Debugging)
            next_t = self._t + 1
            if next_t <= self.T_d:
                next_req = self._instance[next_t - 1]
                self._state = self._build_state(next_t, new_caps, next_req)
            obs = np.array(self._state, dtype=np.float32)
            info = {"valid_actions": []}
            return obs, reward, terminated, truncated, info

        # --- Nächsten Zeitschritt aufbauen ---
        self._t += 1
        self._capacities = new_caps

        if self._t > self.T_d:
            # Episode beendet
            terminated = True
            obs = np.array(state, dtype=np.float32)   # letzter gültiger Zustand
            info = {"valid_actions": []}
        else:
            next_req = self._instance[self._t - 1]
            self._state = self._build_state(self._t, self._capacities, next_req)
            obs = np.array(self._state, dtype=np.float32)
            info = {"valid_actions": self._get_valid_actions(self._state)}

        return obs, reward, terminated, truncated, info

    def render(self):
        """ Einfaches Text-Rendering des aktuellen Zustands """
        if self.render_mode == "human":
            K = self.K
            s = self._state
            print(
                f"t={int(s[0]):<4} "
                f"Caps={[int(c) for c in s[1:K+1]]}  "
                f"r={s[K+1]:.2f}  "
                f"q={[int(q) for q in s[K+2:]]}  "
                f"Valid Actions: {self._get_valid_actions(s)}"
            )

    def get_valid_actions(self):
        """Öffentliche Methode: gibt gültige Aktionen des aktuellen Zustands zurück."""
        return self._get_valid_actions(self._state)


# -----------------------------------------------------------------------
# Schnelltest
# -----------------------------------------------------------------------
if __name__ == "__main__":
    env = DrauspEnv(K=5, T_d=20, C_k=[5] * 5, render_mode="human")
    obs, info = env.reset()
    print("Initialer Zustand:", obs)
    print("Gültige Aktionen:", info["valid_actions"])

    done = False
    total_reward = 0
    while not done:
        valid = env.get_valid_actions()
        action = np.random.choice(valid)          # zufällige gültige Aktion
        print(f"Aktion gewählt: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated

    print(f"Episode beendet. Gesamtreward: {total_reward:.2f}")

