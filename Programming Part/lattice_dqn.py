import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation

class LatticeDQNNetwork(nn.Module):
    """
    DQN mit Monotonie-Garantie.
    Input-Struktur erwartet: [t, C_1...C_K, r, q_1...q_K]
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Berechne K aus input_dim (input_dim = 2K + 2)
        # Bsp: input_dim 12 -> K = 5
        K = (input_dim - 2) // 2
        
        # Konfiguration der Monotonie (DECREASING für t und q_k, INCREASING für C_k und r)
        # Reihenfolge: [t (1), C (K), r (1), q (K)]
        self.mono_configs = (
            [Monotonicity.DECREASING] +      # t
            [Monotonicity.INCREASING] * K +  # C_k
            [Monotonicity.INCREASING] +      # r
            [Monotonicity.DECREASING] * K    # q_k
        )

        # 1. Kalibrierung pro Dimension, mappt alle werte in den Wertebereich des lattice
        self.calibrators = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(0.0, 1.0, 8),
                monotonicity=m,
                output_min=0.0,
                output_max=1.0,
            ) for m in self.mono_configs
        ])

        # 2. Das Lattice (Gitter)
        # units=output_dim sorgt dafür, dass wir direkt Q-Werte für alle Aktionen bekommen
        self.lattice = Lattice(
            lattice_sizes=[2] * input_dim,
            monotonicities=[Monotonicity.INCREASING] * input_dim,
            interpolation=Interpolation.HYPERCUBE,
            units=output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Jeden Input-Kanal einzeln kalibrieren
        calibrated = torch.cat(
            [cal(x[:, i:i+1]) for i, cal in enumerate(self.calibrators)],
            dim=1,
        )
        # Direkt die Q-Werte aus dem Lattice
        return self.lattice(calibrated)

    def apply_constraints(self):
        """Muss nach optimizer.step() aufgerufen werden!
        stellt sicher, dass die Ecken der Lattice Flächen die Monotonie erfüllen"""
        for cal in self.calibrators:
            cal.apply_constraints()
        self.lattice.apply_constraints()