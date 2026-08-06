import torch
import torch.nn as nn
import numpy as np
from itertools import product
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation


class FullLatticeNetwork(nn.Module):
    """
    Zweischichtiges Lattice-Netzwerk für Q(s,a).

    State: [t, C_1..C_K, r, q_1..q_K]

    Architektur:
        1. NumericalCalibrator pro Feature → [0,1]
        2. Schicht 1: Ensemble kleiner 3-Input-Lattices (Gruppen A, B, C)
        3. NumericalCalibrator pro Lattice-Output (Schicht 1 → Schicht 2) → [0,1]
        4. Schicht 2: Cross-resource 3-Input-Lattices (mischen Gruppen und k)
        5. Linear(weights >= 0) → Q pro Aktion

    Monotonie-Annahmen:
        t   : DECREASING (mehr Zeit verbraucht -> schlechter)
        C_k : INCREASING (mehr Kapazität -> besser)
        r   : INCREASING (höherer Reward -> besser)
        q_k : DECREASING (höherer Bedarf -> schlechter)

    Nach Calibrierung in Schicht 1 sind alle Outputs in [0,1] und implizit INCREASING
    (Richtung ist in den Calibratoren absorbiert). Schicht 2 verwendet daher nur
    INCREASING Monotonicities.
    """

    def __init__(
        self,
        input_dim:      int,
        output_dim:     int,
        c_range:        tuple = (0.0,  20.0),
        t_range:        tuple = (1.0,  20.0),
        r_range:        tuple = (0.0, 100.0),
        q_range:        tuple = (0.0,  20.0),
        keypoints:      int   = 8,
        lattice_units:  int   = 4,
        lattice_units2: int   = 2,
    ):
        super().__init__()
        self.K = K = (input_dim - 2) // 2
        self.output_dim = output_dim
        self.lattice_units  = lattice_units
        self.lattice_units2 = lattice_units2

        # ------------------------------------------------------------------
        # 1. Calibratoren (Input → [0,1])
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
        # 2. Schicht 1: Lattice-Gruppen (je 3 Inputs, 2^3 = 8 Gitterpunkte)
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
        # 3. Zwischen-Calibratoren: ein eigener pro Lattice-Output aus Schicht 1
        #
        # Schicht-1 produziert pro k: lattice_units Outputs für A, B, C
        # → 3 * K * lattice_units Calibratoren insgesamt
        # Alle INCREASING, da die Monotonie-Richtung bereits in Schicht 1 kodiert ist
        # ------------------------------------------------------------------
        # Anzahl Outputs Schicht 1: 3 Gruppen * K Ressourcen * lattice_units
        n_l1_outputs = 3 * K * lattice_units
        self.cal_between = nn.ModuleList([
            NumericalCalibrator(
                input_keypoints=np.linspace(0.0, 1.0, keypoints),
                monotonicity=Monotonicity.INCREASING,
                output_min=0.0, output_max=1.0,
            )
            for _ in range(n_l1_outputs)
        ])

        # ------------------------------------------------------------------
        # 4. Schicht 2: Cross-resource Lattices
        #
        # Jedes Lattice bekommt 3 Inputs aus verschiedenen (Gruppe, k)-Kombinationen.
        # Wir erzeugen alle (Gruppe, k, unit)-Indizes und gruppieren sie in 3er-Tripel,
        # die jeweils verschiedene Gruppen UND verschiedene k mischen.
        #
        # Index-Schema für cal_between / Schicht-1-Outputs:
        #   out_A[k][u] → Index: 0*K*U + k*U + u
        #   out_B[k][u] → Index: 1*K*U + k*U + u
        #   out_C[k][u] → Index: 2*K*U + k*U + u
        #
        # Cross-resource Tripel: (Gruppe_i != Gruppe_j != Gruppe_l) und (k_i != k_j wenn möglich)
        # Bei K=1: nur verschiedene Gruppen möglich (k muss gleich sein)
        # ------------------------------------------------------------------
        self.l2_triplets = self._build_cross_resource_triplets(K, lattice_units)
        n_l2_lattices = len(self.l2_triplets)

        mono_L2 = [Monotonicity.INCREASING] * 3  # Richtung steckt bereits in cal_between
        self.lattices_L2 = nn.ModuleList([
            nn.ModuleList([
                Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_L2,
                        interpolation=Interpolation.HYPERCUBE, units=1)
                for _ in range(lattice_units2)
            ])
            for _ in range(n_l2_lattices)
        ])

        # ------------------------------------------------------------------
        # 5. Output-Layer mit nicht-negativen Gewichten
        # ------------------------------------------------------------------
        lattice2_out_dim = n_l2_lattices * lattice_units2
        self.output_layer = nn.Linear(lattice2_out_dim, output_dim)

        with torch.no_grad():
            nn.init.uniform_(self.output_layer.weight, 0.0, 1.0 / lattice2_out_dim)
            nn.init.zeros_(self.output_layer.bias)

    # ----------------------------------------------------------------------

    @staticmethod
    def _build_cross_resource_triplets(K: int, U: int) -> list:
        """
        Baut Tripel von Schicht-1-Output-Indizes für die Cross-resource Schicht 2.

        Jedes Tripel = (idx_A, idx_B, idx_C) mit:
          - idx_A aus Gruppe A, idx_B aus Gruppe B, idx_C aus Gruppe C
          - k-Werte möglichst verschieden (cross-resource)
          - u-Werte variieren um verschiedene Ensemble-Member zu mischen

        Index-Formel:
          Gruppe g ∈ {0,1,2}, Ressource k ∈ {0..K-1}, Unit u ∈ {0..U-1}
          → flat_index = g * K * U + k * U + u
        """
        def idx(g, k, u):
            return g * K * U + k * U + u

        triplets = []
        # Für jede Kombination aus (k_A, k_B, k_C) mit möglichst verschiedenen k
        # und rotierenden u-Werten
        k_combos = list(product(range(K), repeat=3))
        # Bevorzuge Combos wo alle k verschieden (cross-resource)
        k_combos_cross = [c for c in k_combos if len(set(c)) == K] or k_combos

        for u_offset, (kA, kB, kC) in enumerate(k_combos_cross):
            uA = u_offset % U
            uB = (u_offset + 1) % U
            uC = (u_offset + 2) % U
            triplet = (idx(0, kA, uA), idx(1, kB, uB), idx(2, kC, uC))
            triplets.append(triplet)

        return triplets

    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K
        U = self.lattice_units

        # Features extrahieren
        t = x[:, 0:1]
        c = x[:, 1:K+1]
        r = x[:, K+1:K+2]
        q = x[:, K+2:2*K+2]

        # Kalibrieren (Schicht 1 Input)
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

        # ------------------------------------------------------------------
        # Schicht 1: Lattice-Gruppen auswerten
        # Reihenfolge der Outputs: alle A-Outputs (k=0..K-1), dann B, dann C
        # Innerhalb jedes k: lattice_units viele Outputs
        # → Flat-Vektor der Länge 3*K*U für cal_between
        # ------------------------------------------------------------------
        out_A, out_B, out_C = [], [], []
        for k in range(K):
            ck = c_cal[:, k:k+1]
            qk = q_cal[:, k:k+1]

            inp_A = torch.cat([t_cal, r_cal, ck], dim=1).double()
            inp_B = torch.cat([ck, qk, r_cal],    dim=1).double()
            inp_C = torch.cat([t_cal, ck, qk],    dim=1).double()

            # (B, lattice_units) pro Gruppe und k
            out_A.append(torch.cat([lat(inp_A).float() for lat in self.lattices_A[k]], dim=1))
            out_B.append(torch.cat([lat(inp_B).float() for lat in self.lattices_B[k]], dim=1))
            out_C.append(torch.cat([lat(inp_C).float() for lat in self.lattices_C[k]], dim=1))

        # Flat-Vektor: (B, 3*K*U)
        # Reihenfolge: [A_k0_u0..u3, A_k1_u0..u3, B_k0..., C_k0...]
        l1_flat = torch.cat(out_A + out_B + out_C, dim=1)

        # ------------------------------------------------------------------
        # Zwischen-Calibratoren: jeder Output bekommt seinen eigenen Calibrator
        # ------------------------------------------------------------------
        l1_recal = torch.cat(
            [cal(l1_flat[:, i:i+1].double()).float() for i, cal in enumerate(self.cal_between)],
            dim=1,
        )  # (B, 3*K*U)

        # ------------------------------------------------------------------
        # Schicht 2: Cross-resource Lattices
        # Jedes Tripel (idxA, idxB, idxC) greift auf drei kalibrierte L1-Outputs zu
        # ------------------------------------------------------------------
        l2_outputs = []
        for trip_idx, (iA, iB, iC) in enumerate(self.l2_triplets):
            inp_L2 = torch.cat([
                l1_recal[:, iA:iA+1],
                l1_recal[:, iB:iB+1],
                l1_recal[:, iC:iC+1],
            ], dim=1).double()

            # (B, lattice_units2)
            l2_outputs.append(
                torch.cat([lat(inp_L2).float() for lat in self.lattices_L2[trip_idx]], dim=1)
            )

        # Alles zusammenfügen → (B, n_triplets * lattice_units2)
        combined = torch.cat(l2_outputs, dim=1)

        # Output → (B, output_dim)
        return self.output_layer(combined)

    # ----------------------------------------------------------------------

    def apply_constraints(self):
        # Input-Calibratoren
        self.cal_t.apply_constraints()
        self.cal_r.apply_constraints()
        for cal in self.cal_c:
            cal.apply_constraints()
        for cal in self.cal_q:
            cal.apply_constraints()

        # Schicht 1 Lattices
        for group in self.lattices_A:
            for lat in group:
                lat.apply_constraints()
        for group in self.lattices_B:
            for lat in group:
                lat.apply_constraints()
        for group in self.lattices_C:
            for lat in group:
                lat.apply_constraints()

        # Zwischen-Calibratoren
        for cal in self.cal_between:
            cal.apply_constraints()

        # Schicht 2 Lattices
        for group in self.lattices_L2:
            for lat in group:
                lat.apply_constraints()

        # Nicht-negative Output-Gewichte
        with torch.no_grad():
            self.output_layer.weight.clamp_(min=0.0)
            
            
            
# =============================================================================
# ARCHITEKTUR: FullLatticeNetwork
# =============================================================================
#
# Eingabe-State: x = [t, C_1..C_K, r, q_1..q_K]   (Dimension: 2 + 2K)
#
# MONOTONIE-ANNAHMEN (domänenspezifisch):
#   t   DECREASING  — mehr Zeit verbraucht  → schlechtere Situation
#   C_k INCREASING  — mehr Restkapazität    → besser
#   r   INCREASING  — höherer Reward        → besser
#   q_k DECREASING  — höherer Bedarf        → schlechter (schwerer zu erfüllen)
#
# -----------------------------------------------------------------------------
# SCHICHT 0 — Input-Calibratoren  (4 Typen, je K oder 1 Stück)
# -----------------------------------------------------------------------------
#
#   Jedes Raw-Feature wird durch einen NumericalCalibrator auf [0,1] gebracht.
#   Der Calibrator lernt eine stückweise lineare Funktion mit P Keypoints.
#   Die Monotonie-Richtung wird hier ein für alle Mal kodiert, sodass
#   alle nachgelagerten Schichten nur noch INCREASING arbeiten müssen.
#
#   cal_t  : t   → [0,1]  (DECREASING: großes t → kleiner Output)
#   cal_c_k: C_k → [0,1]  (INCREASING: große Kapazität → großer Output)
#   cal_r  : r   → [0,1]  (INCREASING)
#   cal_q_k: q_k → [0,1]  (DECREASING: hoher Bedarf → kleiner Output)
#
#   Parameter pro Calibrator: P (Keypoint-Ausgabewerte)
#   Gesamt: (2 + 2K) * P
#
# -----------------------------------------------------------------------------
# SCHICHT 1 — Ensemble kleiner 3-Input-Lattices  (pro Ressource k)
# -----------------------------------------------------------------------------
#
#   Für jede Ressource k werden drei inhaltlich motivierte Gruppen gebildet,
#   die je drei kalibrierte Features zu einem Lattice zusammenfassen.
#   Ein Lattice mit Größe [2,2,2] hat 2^3 = 8 lernbare Eckgewichte und
#   interpoliert trilinear für Inputs in [0,1]^3.
#
#   Gruppe A: [t_cal, r_cal, C_k_cal]
#     → Beziehung zwischen Zeitdruck, Reward und Kapazität
#   Gruppe B: [C_k_cal, q_k_cal, r_cal]
#     → Beziehung zwischen Kapazität, Bedarf und Reward
#   Gruppe C: [t_cal, C_k_cal, q_k_cal]
#     → Beziehung zwischen Zeitdruck, Kapazität und Bedarf
#
#   Pro Gruppe und k gibt es `lattice_units` unabhängige Lattices (Ensemble),
#   die denselben Input bekommen aber verschiedene Funktionen lernen können.
#   lattice_units := Anzahl der Lattice Blöcke pro Gruppe und Ressource k (z.B. 4)
#
#   Ausgabe pro k: 3 * lattice_units Skalare  ∈ ℝ  (noch nicht in [0,1])
#   Gesamtausgabe Schicht 1: 3 * K * lattice_units Skalare
#   Parameter: 3 * K * lattice_units * 8
#
#   Flacher Index eines Outputs:
#     Gruppe g ∈ {0=A, 1=B, 2=C}, Ressource k, Unit u
#     → flat_idx = g * K * U + k * U + u
#
# -----------------------------------------------------------------------------
# SCHICHT 1→2 — Zwischen-Calibratoren  (ein eigener pro L1-Output)
# -----------------------------------------------------------------------------
#
#   Jeder der 3*K*U Schicht-1-Outputs bekommt einen eigenen INCREASING
#   NumericalCalibrator mit Keypoints in [0,1].
#   Dieser re-normalisiert den Output zurück auf [0,1], damit Schicht 2
#   ihn als gültige Gitterkoordinate verwenden kann.
#
#   Da die Monotonie-Richtung bereits durch die Input-Calibratoren kodiert
#   ist, genügt INCREASING für alle Zwischen-Calibratoren.
#
#   Parameter: 3 * K * U * P
#
# -----------------------------------------------------------------------------
# SCHICHT 2 — Cross-resource Lattices
# -----------------------------------------------------------------------------
#
#   Während Schicht 1 nur intra-resource Interaktionen lernt
#   (alle 3 Inputs eines Lattices gehören zur selben Ressource k),
#   mischen die Schicht-2-Lattices bewusst Outputs verschiedener
#   Gruppen AND verschiedener Ressourcen.
#
#   Jedes Tripel hat die Form (A_kA_uA, B_kB_uB, C_kC_uC):
#     — immer eine aus jeder Gruppe (A, B, C)
#     — möglichst kA ≠ kB ≠ kC  (cross-resource)
#     — rotierende Unit-Indizes (verschiedene Ensemble-Member)
#
#   Dadurch kann Schicht 2 lernen, wie z.B. die Kapazitätssituation
#   von Ressource 1 mit dem Zeitdruck-Reward-Profil von Ressource 2
#   interagiert — was Schicht 1 strukturell nicht modellieren kann.
#
#   Alle Lattices in Schicht 2 sind INCREASING (Richtung steckt in den
#   Input-Calibratoren und wird durch INCREASING Zwischen-Calibratoren
#   durchgereicht).
#
#   Ausgabe: n_triplets * lattice_units2 Skalare
#   Parameter: n_triplets * lattice_units2 * 8
#
# -----------------------------------------------------------------------------
# SCHICHT 3 — Output-Layer  Linear(≥0) → Q(s,a)
# -----------------------------------------------------------------------------
#
#   Ein einfacher Linear-Layer aggregiert alle Schicht-2-Outputs zu einem
#   Q-Wert pro Aktion. Die Gewichte werden auf ≥ 0 geclampt (in
#   apply_constraints()), sodass die end-to-end Monotonie erhalten bleibt:
#   ein positiver Einfluss in Schicht 1 kann durch den Output-Layer nicht
#   umgekehrt werden.
#
#   Parameter: (n_triplets * lattice_units2 + 1) * output_dim
#
# -----------------------------------------------------------------------------
# PARAMETERÜBERSICHT (Beispiel K=2, P=8, U=4, U2=2, D_out=2)
# -----------------------------------------------------------------------------
#
#   Input-Calibratoren  (2+2*2)*8         =  48
#   Lattice Schicht 1   3*2*4*8           = 192
#   Zwischen-Calibrat.  3*2*4*8           = 192
#   Lattice Schicht 2   8 Triplets*2*8    = 128
#   Output-Layer        (16+1)*2          =  34
#                                    Σ  = 594
#
# =============================================================================