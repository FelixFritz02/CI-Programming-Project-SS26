import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation


class FullLatticeNetwork(nn.Module):
    """
    Reines Lattice-Netzwerk für Q(s,a).

    State: [t, C_1..C_K, r, q_1..q_K]

    Architektur:
        1. NumericalCalibrator pro Feature → [0,1]
        2. Ensemble kleiner 3-Input-Lattices (Gruppen A, B, C)
        3. Linear(weights >= 0) → Q pro Aktion

    Monotonie-Annahmen:
        t   : DECREASING (mehr Zeit verbraucht -> schlechter)
        C_k : INCREASING (mehr Kapazität -> besser)
        r   : INCREASING (höherer Reward -> besser)
        q_k : DECREASING (höherer Bedarf -> schlechter)
    """

    def __init__(
        self,
        input_dim:    int,
        output_dim:   int,
        c_range:      tuple = (0.0,  20.0),
        t_range:      tuple = (1.0,  20.0),
        r_range:      tuple = (0.0, 100.0),
        q_range:      tuple = (0.0,  20.0),
        keypoints:    int   = 8,
        lattice_units: int  = 4,
    ):
        super().__init__()
        self.K = K = (input_dim - 2) // 2
        self.output_dim = output_dim

        # ------------------------------------------------------------------
        # 1. Calibratoren
        # ------------------------------------------------------------------
        self.cal_t = NumericalCalibrator(
            input_keypoints=np.linspace(t_range[0], t_range[1], keypoints),
            monotonicity=Monotonicity.DECREASING,
            output_min=0.0, output_max=1.0,
        )
        self.cal_c = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(c_range[0], c_range[1], keypoints),
                monotonicity=Monotonicity.INCREASING,
                output_min=0.0, output_max=1.0,
            )
            for _ in range(K)
        ])
        self.cal_r = NumericalCalibrator(
            input_keypoints=np.linspace(r_range[0], r_range[1], keypoints),
            monotonicity=Monotonicity.INCREASING,
            output_min=0.0, output_max=1.0,
        )
        self.cal_q = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(q_range[0], q_range[1], keypoints),
                monotonicity=Monotonicity.DECREASING,
                output_min=0.0, output_max=1.0,
            )
            for _ in range(K)
        ])

        # ------------------------------------------------------------------
        # 2. Lattice-Gruppen (je 3 Inputs, 2^3 = 8 Gitterpunkte)
        #
        # Gruppe A: [t↓, r↑, C_k↑]
        # Gruppe B: [C_k↑, q_k↓, r↑]
        # Gruppe C: [t↓, C_k↑, q_k↓]
        # ------------------------------------------------------------------
        mono_A = [Monotonicity.DECREASING, Monotonicity.INCREASING, Monotonicity.INCREASING]
        mono_B = [Monotonicity.INCREASING, Monotonicity.DECREASING, Monotonicity.INCREASING]
        mono_C = [Monotonicity.DECREASING, Monotonicity.INCREASING, Monotonicity.DECREASING]

        self.lattices_A = nn.ModuleList([
            nn.ModuleList([
                Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_A,
                        interpolation=Interpolation.HYPERCUBE, units=1)
                for _ in range(lattice_units)
            ])
            for _ in range(K)
        ])
        self.lattices_B = nn.ModuleList([
            nn.ModuleList([
                Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_B,
                        interpolation=Interpolation.HYPERCUBE, units=1)
                for _ in range(lattice_units)
            ])
            for _ in range(K)
        ])
        self.lattices_C = nn.ModuleList([
            nn.ModuleList([
                Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_C,
                        interpolation=Interpolation.HYPERCUBE, units=1)
                for _ in range(lattice_units)
            ])
            for _ in range(K)
        ])

        # ------------------------------------------------------------------
        # 3. Output-Layer mit nicht-negativen Gewichten
        #    → Monotonie bleibt durch die gesamte Architektur erhalten
        # ------------------------------------------------------------------
        lattice_out_dim = 3 * K * lattice_units
        self.output_layer = nn.Linear(lattice_out_dim, output_dim)

        with torch.no_grad():
            nn.init.uniform_(self.output_layer.weight, 0.0, 1.0 / lattice_out_dim)
            nn.init.zeros_(self.output_layer.bias)

    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K

        # Features extrahieren
        t = x[:, 0:1]
        c = x[:, 1:K+1]
        r = x[:, K+1:K+2]
        q = x[:, K+2:2*K+2]

        # Kalibrieren
        t_cal = self.cal_t(t.double()).float()
        r_cal = self.cal_r(r.double()).float()
        c_cal = torch.cat(
            [cal(c[:, k:k+1].double()).float() for k, cal in enumerate(self.cal_c)],
            dim=1,
        )
        q_cal = torch.cat(
            [cal(q[:, k:k+1].double()).float() for k, cal in enumerate(self.cal_q)],
            dim=1,
        )

        # Lattice-Gruppen auswerten
        out_A, out_B, out_C = [], [], []
        for k in range(K):
            ck = c_cal[:, k:k+1]
            qk = q_cal[:, k:k+1]

            inp_A = torch.cat([t_cal, r_cal, ck], dim=1).double()
            inp_B = torch.cat([ck, qk, r_cal],    dim=1).double()
            inp_C = torch.cat([t_cal, ck, qk],    dim=1).double()

            # Alle units zusammenfügen → (B, lattice_units)
            out_A.append(torch.cat([lat(inp_A).float() for lat in self.lattices_A[k]], dim=1))
            out_B.append(torch.cat([lat(inp_B).float() for lat in self.lattices_B[k]], dim=1))
            out_C.append(torch.cat([lat(inp_C).float() for lat in self.lattices_C[k]], dim=1))

        # Alles zusammenfügen → (B, 3·K·units)
        combined = torch.cat(out_A + out_B + out_C, dim=1)

        # Output → (B, output_dim)
        return self.output_layer(combined)

    # ----------------------------------------------------------------------

    def apply_constraints(self):
        self.cal_t.apply_constraints()
        self.cal_r.apply_constraints()
        for cal in self.cal_c:
            cal.apply_constraints()
        for cal in self.cal_q:
            cal.apply_constraints()
        for group in self.lattices_A:
            for lat in group:
                lat.apply_constraints()
        for group in self.lattices_B:
            for lat in group:
                lat.apply_constraints()
        for group in self.lattices_C:
            for lat in group:
                lat.apply_constraints()
        # Nicht-negative Gewichte erzwingen
        with torch.no_grad():
            self.output_layer.weight.clamp_(min=0.0)