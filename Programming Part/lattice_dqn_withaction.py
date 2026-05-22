import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation


class LatticeDQNNetwork(nn.Module):
    """
    DQN + aktionsabhängiges monotones Lattice.

    Architektur:

        Pfad A – normales DQN:
            [t, C_1..C_K, r, q_1..q_K]
                → MLP
                → Q_dqn(s,a)

        Pfad B – monotones Lattice:
            Für jede Aktion a:
                [C_1..C_K, a]
                    → Calibrators für C_k
                    → Lattice
                    → Q_lattice(s,a)

        Output:
            Q(s,a) = Q_dqn(s,a) + Q_lattice(s,a)

    WICHTIG:
        - Monotonie nur bzgl. C_k
        - NICHT bzgl. der Aktion
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        c_range: tuple = (0.0, 10.0),
        keypoints: int = 8,
    ):
        super().__init__()

        self.K = K = (input_dim - 2) // 2
        self.output_dim = output_dim

        # --------------------------------------------------------------
        # Pfad A: Standard DQN
        # --------------------------------------------------------------
        self.dqn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        # --------------------------------------------------------------
        # Pfad B: Kalibratoren für C_k
        # --------------------------------------------------------------
        self.c_calibrators = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(c_range[0], c_range[1], keypoints),
                monotonicity=Monotonicity.INCREASING,
                output_min=0.0,
                output_max=1.0,
            )
            for _ in range(K)
        ])

        # --------------------------------------------------------------
        # Aktionskalibrator
        # --------------------------------------------------------------
        self.action_calibrator = NumericalCalibrator(
            input_keypoints=np.linspace(0, output_dim - 1, output_dim),
            monotonicity=None,   # WICHTIG: keine Monotonie für Aktionen
            output_min=0.0,
            output_max=1.0,
        )

        # --------------------------------------------------------------
        # Lattice:
        # Inputs:
        #   C_1,...,C_K,a
        # --------------------------------------------------------------
        self.c_lattice = Lattice(
            lattice_sizes=[2] * (K + 1),
            monotonicities=
                [Monotonicity.INCREASING] * K
                + [None],   # Aktion NICHT monoton
            interpolation=Interpolation.HYPERCUBE,
            units=1,
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        K = self.K
        A = self.output_dim

        # --------------------------------------------------------------
        # Pfad A: normales DQN
        # --------------------------------------------------------------
        q_dqn = self.dqn(x)   # (B, A)

        # --------------------------------------------------------------
        # C_k extrahieren
        # --------------------------------------------------------------
        c_raw = x[:, 1:K+1]   # (B, K)

        # --------------------------------------------------------------
        # C_k kalibrieren
        # --------------------------------------------------------------
        c_cal = torch.cat(
            [
                cal(c_raw[:, k:k+1].double()).float()
                for k, cal in enumerate(self.c_calibrators)
            ],
            dim=1,
        )  # (B, K)

        # --------------------------------------------------------------
        # Für jede Aktion eigenes Lattice evaluieren
        # --------------------------------------------------------------
        q_lattice_list = []

        for a in range(A):

            # Aktionstensor
            a_tensor = torch.full(
                (B, 1),
                float(a),
                dtype=torch.float32,
                device=x.device,
            )

            # Aktion kalibrieren
            a_cal = self.action_calibrator(
                a_tensor.double()
            ).float()  # (B,1)

            # Input:
            # [C_1,...,C_K,a]
            lattice_input = torch.cat(
                [c_cal, a_cal],
                dim=1,
            )  # (B, K+1)

            # Lattice auswerten
            q_lat_a = self.c_lattice(
                lattice_input.double()
            ).float()  # (B,1)

            q_lattice_list.append(q_lat_a)

        # Alle Aktionen zusammenfügen
        #Bonus für jede Ation in einem Tensor
        q_lattice = torch.cat(q_lattice_list, dim=1)  # (B, A)

        # --------------------------------------------------------------
        # Finale Q-Werte
        # --------------------------------------------------------------
        return q_dqn + q_lattice

    # ------------------------------------------------------------------

    def apply_constraints(self):

        for cal in self.c_calibrators:
            cal.apply_constraints()

        self.action_calibrator.apply_constraints()

        self.c_lattice.apply_constraints()