import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import Lattice, NumericalCalibrator
from pytorch_lattice.enums import Monotonicity, Interpolation

class LatticeDQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Konfiguration der Monotonie 
        K = (input_dim - 2) // 2
        self.mono_configs = (
            [None] +    # Zeit t
            [Monotonicity.INCREASING] * K + # Kapazitäten C_k
            [None] +    # Erlös r
            [None] * K   # Benötigte Kapazität q_k
        )

        # kalibirieren der input auf 0,1 skala mit monotonem stückweise linearem Kalibrator
        self.calibrators = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(0.0, 1.0, 8),
                monotonicity=m,
                output_min=0.0,
                output_max=1.0,
            ) for m in self.mono_configs
        ])

        # 2. Workaround: Ein eigenes Lattice pro Aktion (unit=1)
        self.lattices = nn.ModuleList([
            Lattice(
                lattice_sizes=[2] * input_dim,
                monotonicities=[Monotonicity.INCREASING] * input_dim,
                interpolation=Interpolation.HYPERCUBE,
                units=1, # Jedes Gitter berechnet nur EINEN Q-Wert
            ) for _ in range(output_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Alle Features kalibrieren
        calibrated = torch.cat(
            [cal(x[:, i:i+1]) for i, cal in enumerate(self.calibrators)],
            dim=1,
        )
        
        # Jedes Lattice einzeln aufrufen und Ergebnisse seitlich zusammenfügen (Batch, output_dim)
        # lat(calibrated) gibt (Batch, 1) zurück
        q_values = torch.cat([lat(calibrated) for lat in self.lattices], dim=1)
        
        return q_values.float()

    def apply_constraints(self):
        # Constraints auf alle Kalibratoren und alle Gitter anwenden
        for cal in self.calibrators:
            cal.apply_constraints()