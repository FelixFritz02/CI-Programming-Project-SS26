import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator
from pytorch_lattice.enums import Monotonicity


import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation


class LatticeDQNNetwork(nn.Module):
    """
    DQN + Lattice für C_k-Monotonie.

    Architektur:

        Pfad A – alle Features durch normales DQN:
            [t, C_1..C_K, r, q_1..q_K] → Linear(256) → ReLU → Linear(256)
            → ReLU → Linear(128) → ReLU → Linear(output_dim)
            → Q_dqn  (B, output_dim)

        Pfad B – nur C_k durch Lattice:
            [C_1..C_K] → K × NumericalCalibrator(INCREASING) → [0,1]^K
            → Lattice(K Inputs, INCREASING, 2^K Punkte)
            → Q_lattice  (B, 1)   ← ein Skalar, aktionsunabhängig

        Output:
            Q = Q_dqn + Q_lattice    (Broadcasting über Aktionen)

    Warum addieren statt concatenieren?
        Q_lattice ist ein aktionsunabhängiger Bonus/Malus der nur von den
        Kapazitäten abhängt. Das entspricht genau der Semantik: mehr Kapazität
        bedeutet generell bessere Situation, unabhängig von der konkreten Aktion.
        Das DQN lernt die aktionsspezifischen Unterschiede, das Lattice lernt
        den globalen Kapazitäts-Effekt.
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        c_range:    tuple = (0.0,  10.0),
        keypoints:  int   = 8,
    ):
        super().__init__()

        self.K = K = (input_dim - 2) // 2

        # ------------------------------------------------------------------
        # Pfad A: Standard DQN (alle Features, roh)
        # ------------------------------------------------------------------
        self.dqn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        # ------------------------------------------------------------------
        # Pfad B: Lattice nur für C_1..C_K
        # ------------------------------------------------------------------
        # Ein Kalibrator pro C_k, alle INCREASING
        self.c_calibrators = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(c_range[0], c_range[1], keypoints),
                monotonicity=Monotonicity.INCREASING,
                output_min=0.0,
                output_max=1.0,
            )
            for _ in range(K)
        ])

        # K-dimensionales Lattice, alle Achsen INCREASING
        # 2^K Gitterpunkte – bei K=4 sind das 16, bei K=5 sind das 32
        self.c_lattice = Lattice(
            lattice_sizes=[2] * K,
            monotonicities=[Monotonicity.INCREASING] * K,
            interpolation=Interpolation.HYPERCUBE,
            units=1,
        )

    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K

        # Pfad A: DQN mit allen Features → (B, output_dim)
        q_dqn = self.dqn(x)

        # Pfad B: C_k extrahieren (Indizes 1..K im State-Vektor)
        c_raw = x[:, 1:K+1]   # (B, K)

        # Kalibrieren – Kalibrator gibt float64 zurück → float32
        c_cal = torch.cat(
            [cal(c_raw[:, k:k+1].double()).float() for k, cal in enumerate(self.c_calibrators)],
            dim=1,
        )  # (B, K)

        # Lattice → (B, 1), aktionsunabhängiger Kapazitäts-Bonus, für jede Aktion der gleiche Bonus
        q_lattice = self.c_lattice(c_cal.double()).float()  # (B, 1)

        # Addieren: Broadcasting auf (B, output_dim)
        return q_dqn + q_lattice

    def apply_constraints(self):
        for cal in self.c_calibrators:
            cal.apply_constraints()
        self.c_lattice.apply_constraints()
