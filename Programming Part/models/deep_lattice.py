"""
DeepLattice: zweistufiges monotones Lattice-Netzwerk für Q-Learning.

Dieses Modul implementiert ein hierarchisches Lattice-Modell, das zuerst
mehrere 3-Input-Lattices pro Ressource verarbeitet und deren Ausgaben
über Zwischenkalibratoren in eine zweite Cross-Resource-Lattice-Schicht
führt. Am Ende aggregiert ein nicht-negativer Linear-Layer die Features
zu Q-Werten pro Aktion.

Die Architektur nutzt domänenspezifische Monotonieannahmen für Zeit,
Kapazität, Reward und Bedarf und sorgt so für ein strukturiertes,
interpretierbares Modell mit monotonen Beziehungen.
"""
import torch
import torch.nn as nn
import numpy as np
from pytorch_lattice.layers import NumericalCalibrator, Lattice
from pytorch_lattice.enums import Monotonicity, Interpolation


class DeepLatticeNetwork(nn.Module):
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

    Ensemble-Aggregation: Jede Gruppe (A/B/C) und Ressource k wird von einem Lattice
    mit `lattice_units` (U) parallelen Ensemble-Membern ausgewertet; die U Outputs
    werden direkt danach über dim=1 gemittelt (arithmetisches Mittel monotoner
    Funktionen ist selbst monoton). Dadurch fließt die Information aller U Member in
    genau EINEN Wert pro Gruppe/Ressource, statt dass Schicht 2 nur einen einzelnen,
    per Rotation ausgewählten Member sieht und der Rest ungenutzt verworfen wird.

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

        # Jede Ressource k bekommt EIN Lattice mit units=lattice_units statt
        # lattice_units einzelnen Lattices mit units=1.
        #
        # "units" ist kein Modell-Unterschied, nur eine andere Berechnungsart:
        # Ein Lattice mit units=U speichert an jeder der 8 Gitterecken einen
        # Gewichtsvektor der Länge U statt eines einzelnen Gewichts (kernel-
        # Form (8, U) statt (8, 1)) und ist damit intern äquivalent zu U
        # unabhängigen Lattices mit derselben Gitter-Geometrie. Da alle U
        # Ensemble-Member denselben Input bekommen, kann die Bibliothek die
        # Interpolationsgewichte EINMAL berechnen und für alle U Gewichts-
        # vektoren gleichzeitig verwenden (torch.sum(weights * kernel.t())),
        # statt U-mal denselben Interpolationsschritt einzeln zu wiederholen
        # und die U Einzel-Outputs danach mit cat() zusammenzufügen.
        # Ergebnis ist numerisch identisch, aber ein Tensor-Op statt U
        # Python-Aufrufe (siehe forward(): lat(inp.unsqueeze(1))).
        #
        # output_min/output_max = 0/1: Der Schicht-1-Output geht direkt in die
        # Zwischen-Calibratoren, deren Keypoints fest auf [0,1] liegen. Ohne
        # Bound initialisieren die Lattices auf [-2,2] und driften frei; alles
        # außerhalb [0,1] wird von cal_between hart geclippt -> Nullgradient auf
        # die betroffenen Eckgewichte und die vorgelagerten Input-Calibratoren,
        # und Schicht 2 kann verschiedene out-of-range Werte nicht unterscheiden.
        # Mit dem Bound wird der Output per Konstruktion [0,1] (Konvexkombination
        # von Eckgewichten in [0,1]) und liegt damit exakt im interpolierenden
        # Bereich von cal_between. Kein Ausdrucksverlust: cal_between ist ein
        # lernbarer monotoner Warp und absorbiert jede monotone Umskalierung;
        # die Q-Größenordnung tragen Schicht 2 und der Output-Layer (unbeschränkt).
        self.lattices_A = nn.ModuleList([
            Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_A,
                    interpolation=Interpolation.HYPERCUBE, units=lattice_units,
                    output_min=0.0, output_max=1.0)
            for _ in range(K)
        ])
        self.lattices_B = nn.ModuleList([
            Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_B,
                    interpolation=Interpolation.HYPERCUBE, units=lattice_units,
                    output_min=0.0, output_max=1.0)
            for _ in range(K)
        ])
        self.lattices_C = nn.ModuleList([
            Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_C,
                    interpolation=Interpolation.HYPERCUBE, units=lattice_units,
                    output_min=0.0, output_max=1.0)
            for _ in range(K)
        ])

        # ------------------------------------------------------------------
        # 3. Zwischen-Calibratoren: ein eigener pro (Gruppe, Ressource)
        #
        # Schicht-1 produziert pro k und Gruppe lattice_units (U) Ensemble-Outputs,
        # die in forward() bereits über die U Units gemittelt werden, bevor sie
        # hier ankommen → nur noch 3 * K Calibratoren insgesamt (nicht 3*K*U),
        # und jeder davon ist von ALLEN U Ensemble-Membern der jeweiligen
        # Gruppe/Ressource informiert statt nur von einem einzelnen.
        # Alle INCREASING, da die Monotonie-Richtung bereits in Schicht 1 kodiert ist
        # ------------------------------------------------------------------
        # Anzahl Outputs Schicht 1 nach Ensemble-Mittelung: 3 Gruppen * K Ressourcen
        n_l1_outputs = 3 * K
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
        # Jedes Lattice bekommt 3 Inputs: je einen Wert aus Gruppe A, B, C
        # von jeweils möglichst unterschiedlichen Ressourcen k.
        #
        # Index-Schema für cal_between / Schicht-1-Outputs (nach Ensemble-Mittelung
        # ist jede (Gruppe, Ressource)-Kombination bereits genau EIN Wert):
        #   out_A[k] → Index: 0*K + k
        #   out_B[k] → Index: 1*K + k
        #   out_C[k] → Index: 2*K + k
        #
        # _build_cross_resource_triplets erzeugt genau K Tripel per
        # rotierendem Offset (kA=i, kB=i+1, kC=i+2 mod K) statt aller K^3
        # möglichen (kA,kB,kC)-Kombinationen — sonst würde Schicht 2 kubisch
        # statt linear mit K wachsen (bei K=5 wären das 125 statt 5 Lattices).
        # Bei K>=3 sind kA,kB,kC dadurch immer drei verschiedene Ressourcen,
        # bei K<3 die maximal mögliche Anzahl. Da jede (Gruppe, Ressource)-
        # Kombination bereits alle U Ensemble-Member gemittelt enthält, deckt
        # dies automatisch die volle Schicht-1-Information ab — anders als bei
        # einer reinen Index-Auswahl bleibt hier nichts ungenutzt.
        # ------------------------------------------------------------------
        self.l2_triplets = self._build_cross_resource_triplets(K)
        n_l2_lattices = len(self.l2_triplets)
        # Tripel-Indizes als Tensor (K, 3) für die vektorisierte Schicht 2 in
        # forward(). persistent=False -> nicht im state_dict, keine Auswirkung
        # auf gespeicherte Modelle.
        self.register_buffer(
            "_l2_idx", torch.tensor(self.l2_triplets, dtype=torch.long), persistent=False
        )

        mono_L2 = [Monotonicity.INCREASING] * 3  # Richtung steckt bereits in cal_between
        self.lattices_L2 = nn.ModuleList([
            Lattice(lattice_sizes=[2, 2, 2], monotonicities=mono_L2,
                    interpolation=Interpolation.HYPERCUBE, units=lattice_units2)
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
    def _build_cross_resource_triplets(K: int) -> list:
        """
        Baut Tripel von Schicht-1-Output-Indizes für die Cross-resource Schicht 2.

        Jedes Tripel = (idx_A, idx_B, idx_C) mit:
          - idx_A aus Gruppe A, idx_B aus Gruppe B, idx_C aus Gruppe C
          - k-Werte möglichst verschieden (cross-resource): bei K>=3 immer 3
            verschiedene Ressourcen, bei K<3 die maximal mögliche Anzahl

        Erzeugt genau K Tripel (rotierender Offset), nicht alle K^3
        Kombinationen aus (k_A, k_B, k_C) — sonst wächst Schicht 2 kubisch
        mit der Ressourcenzahl K statt linear. Da jede (Gruppe, Ressource)-
        Kombination bereits über alle lattice_units Ensemble-Member gemittelt
        wurde (siehe forward()), wird durch die K Tripel jede Kombination
        genau einmal verwendet — keine Ensemble-Information geht verloren.

        Index-Formel:
          Gruppe g ∈ {0,1,2}, Ressource k ∈ {0..K-1} → flat_index = g * K + k
        """
        def idx(g, k):
            return g * K + k

        triplets = []
        for offset in range(K):
            kA, kB, kC = offset % K, (offset + 1) % K, (offset + 2) % K
            triplets.append((idx(0, kA), idx(1, kB), idx(2, kC)))

        return triplets

    # ----------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _batched_apply_constraints(calibrators: nn.ModuleList) -> None:
        """
        Projiziert alle Calibratoren in `calibrators` in einem Batch-Op statt
        N einzelnen apply_constraints()-Aufrufen mit je 8 Dykstra-Iterationen.

        Setzt voraus, dass alle Calibratoren dieselbe output_min/output_max/
        monotonicity/projection_iterations-Konfiguration teilen (hier immer
        der Fall). Die Projektionsformeln aus NumericalCalibrator arbeiten
        bereits elementweise/über dim=0, daher funktionieren sie unverändert
        auf einem (n_keypoints, N)-Kernel statt (n_keypoints, 1) — es wird
        nur die Iteration über die Calibratoren selbst eingespart.
        """
        ref = calibrators[0]
        constrain_bounds = ref.output_min is not None or ref.output_max is not None
        constrain_monotonicity = ref.monotonicity is not None
        num_constraints = sum([constrain_bounds, constrain_monotonicity])
        if num_constraints == 0:
            return

        kernel = torch.cat([cal.kernel.data for cal in calibrators], dim=1)  # (n_keypoints, N)
        original_bias, original_heights = kernel[0:1], kernel[1:]

        previous_bias_delta = {"BOUNDS": torch.zeros_like(original_bias)}
        previous_heights_delta = {
            "BOUNDS": torch.zeros_like(original_heights),
            "MONOTONICITY": torch.zeros_like(original_heights),
        }

        def apply_bound_constraints(bias, heights):
            previous_bias = bias - previous_bias_delta["BOUNDS"]
            previous_heights = heights - previous_heights_delta["BOUNDS"]
            if constrain_monotonicity:
                bias, heights = ref._project_monotonic_bounds(previous_bias, previous_heights)
            else:
                bias, heights = ref._approximately_project_bounds_only(previous_bias, previous_heights)
            previous_bias_delta["BOUNDS"] = bias - previous_bias
            previous_heights_delta["BOUNDS"] = heights - previous_heights
            return bias, heights

        def apply_monotonicity_constraints(heights):
            previous_heights = heights - previous_heights_delta["MONOTONICITY"]
            heights = ref._project_monotonicity(previous_heights)
            previous_heights_delta["MONOTONICITY"] = heights - previous_heights
            return heights

        def apply_dykstras_projection(bias, heights):
            if constrain_bounds:
                bias, heights = apply_bound_constraints(bias, heights)
            if constrain_monotonicity:
                heights = apply_monotonicity_constraints(heights)
            return bias, heights

        def finalize_constraints(bias, heights):
            if constrain_monotonicity:
                heights = ref._project_monotonicity(heights)
            if constrain_bounds:
                if constrain_monotonicity:
                    bias, heights = ref._squeeze_by_scaling(bias, heights)
                else:
                    bias, heights = ref._approximately_project_bounds_only(bias, heights)
            return bias, heights

        projected_bias, projected_heights = apply_dykstras_projection(
            original_bias, original_heights
        )
        if num_constraints > 1:
            for _ in range(ref.projection_iterations - 1):
                projected_bias, projected_heights = apply_dykstras_projection(
                    projected_bias, projected_heights
                )
            projected_bias, projected_heights = finalize_constraints(
                projected_bias, projected_heights
            )

        new_kernel = torch.cat((projected_bias, projected_heights), 0)  # (n_keypoints, N)
        for i, cal in enumerate(calibrators):
            cal.kernel.data = new_kernel[:, i:i+1].contiguous()

    @staticmethod
    def _batched_calibrate(x: torch.Tensor, calibrators: nn.ModuleList) -> torch.Tensor:
        """
        Kalibriert alle N Kanäle von x (B, N) in einem Batch-Op statt N
        einzelnen NumericalCalibrator-Aufrufen.

        Setzt voraus, dass alle Calibratoren in `calibrators` dieselben
        (fixen) Keypoints benutzen — das ist hier immer der Fall, da jede
        Gruppe (cal_c, cal_q, cal_between) mit identischer Konfiguration
        erzeugt wird. Jeder Calibrator hat aber weiterhin sein eigenes
        lernbares kernel und wird bei apply_constraints() individuell
        projiziert.
        """
        ref = calibrators[0]
        keypoints = ref._interpolation_keypoints  # (n_keypoints-1,)
        lengths = ref._lengths                     # (n_keypoints-1,)
        kernel = torch.cat([cal.kernel for cal in calibrators], dim=1)  # (n_keypoints, N)

        weights = (x.double().unsqueeze(-1) - keypoints) / lengths  # (B, N, n_keypoints-1)
        weights = weights.clamp(0.0, 1.0)
        weights = torch.cat([torch.ones_like(weights[..., :1]), weights], dim=-1)  # (B, N, n_keypoints)

        return torch.einsum("bnk,kn->bn", weights, kernel).float()

    @staticmethod
    def _hypercube_batch(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """Vektorisierte HYPERCUBE-Interpolation für N parallele 3-Input-Lattices
        der Größe [2,2,2] (clip_inputs=True) — numerisch identisch zu N einzelnen
        `pytorch_lattice.Lattice.forward`-Aufrufen, nur ohne Python-Schleife.

        x       : (B, N, 3)  Inputs je Lattice
        kernels : (N, 8, U)  gestapelte Eckgewichte (`Lattice.kernel` je Modul)
        return  : (B, N, U)  interpolierte Werte (float)

        Replikat von `Lattice._compute_hypercube_interpolation`:
        pro Achse i die zwei Gewichte [1-x_i, x_i] auf [0,1] geclippt, flacher
        Outer-Product in C-Reihenfolge (corner = c0*4 + c1*2 + c2), dann
        sum_corner iw * kernel — alles in float64 wie in der Bibliothek.
        """
        x = x.double()
        xi = x.unsqueeze(-1)                                     # (B, N, 3, 1)
        w = torch.cat([1.0 - xi, xi], dim=-1).clamp(0.0, 1.0)    # (B, N, 3, 2)
        w0, w1, w2 = w[..., 0, :], w[..., 1, :], w[..., 2, :]    # je (B, N, 2)
        iw = (w0[..., :, None, None]
              * w1[..., None, :, None]
              * w2[..., None, None, :]).reshape(x.shape[0], x.shape[1], 8)
        out = torch.einsum("bnc,ncu->bnu", iw, kernels.double())
        return out.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.K

        # Features extrahieren
        t = x[:, 0:1]
        c = x[:, 1:K+1]
        r = x[:, K+1:K+2]
        q = x[:, K+2:2*K+2]

        # Kalibrieren (Schicht 1 Input)
        t_cal = self.cal_t(t.double()).float()          # (B, 1)
        r_cal = self.cal_r(r.double()).float()          # (B, 1)
        c_cal = self._batched_calibrate(c, self.cal_c)  # (B, K)
        q_cal = self._batched_calibrate(q, self.cal_q)  # (B, K)

        # ------------------------------------------------------------------
        # Schicht 1 (vektorisiert):
        # Statt `for k in range(K)` mit je 3 Einzel-Lattice-Aufrufen werden
        # alle K Ressourcen einer Gruppe in EINEM Op ausgewertet. Die K
        # Lattice-Kernels der ModuleList werden dafür gestapelt — exakt das
        # Muster von `_batched_calibrate` für die Calibratoren. Parameter,
        # Monotonie-Richtungen und `apply_constraints()` bleiben unverändert;
        # jede (Gruppe, Ressource) hat weiterhin ihr eigenes Lattice mit U
        # Ensemble-Membern, die danach über dim=-1 gemittelt werden.
        # Reihenfolge der Outputs: A (k=0..K-1), dann B, dann C  → (B, 3K),
        # Flat-Index = g*K + k (wie `_build_cross_resource_triplets`).
        # ------------------------------------------------------------------
        tK = t_cal.expand(-1, K)                                 # (B, K)
        rK = r_cal.expand(-1, K)

        inp_A = torch.stack([tK, rK, c_cal], dim=-1)             # (B, K, 3)  [t, r, C_k]
        inp_B = torch.stack([c_cal, q_cal, rK], dim=-1)          # (B, K, 3)  [C_k, q_k, r]
        inp_C = torch.stack([tK, c_cal, q_cal], dim=-1)          # (B, K, 3)  [t, C_k, q_k]

        W_A = torch.stack([lat.kernel for lat in self.lattices_A], dim=0)  # (K, 8, U)
        W_B = torch.stack([lat.kernel for lat in self.lattices_B], dim=0)
        W_C = torch.stack([lat.kernel for lat in self.lattices_C], dim=0)

        out_A = self._hypercube_batch(inp_A, W_A).mean(dim=2)    # (B, K)  Ensemble-Mittel über U
        out_B = self._hypercube_batch(inp_B, W_B).mean(dim=2)
        out_C = self._hypercube_batch(inp_C, W_C).mean(dim=2)

        l1_flat = torch.cat([out_A, out_B, out_C], dim=1)        # (B, 3K)

        # ------------------------------------------------------------------
        # Zwischen-Calibratoren: jeder der 3K Outputs hat seinen eigenen
        # ------------------------------------------------------------------
        l1_recal = self._batched_calibrate(l1_flat, self.cal_between)  # (B, 3K)

        # ------------------------------------------------------------------
        # Schicht 2 (vektorisiert): die K Cross-Resource-Tripel gleichzeitig.
        # `_l2_idx` (K, 3) sind die Flat-Indizes (iA, iB, iC) in l1_recal.
        # Kein Ensemble-Mittel hier (wie bisher) → (B, K*U2).
        # ------------------------------------------------------------------
        X_L2 = l1_recal[:, self._l2_idx]                         # (B, K, 3)
        W_L2 = torch.stack([lat.kernel for lat in self.lattices_L2], dim=0)  # (K, 8, U2)
        l2 = self._hypercube_batch(X_L2, W_L2)                   # (B, K, U2)
        combined = l2.reshape(l2.shape[0], -1)                   # (B, K*U2)

        # Output → (B, output_dim)
        return self.output_layer(combined)

    # ----------------------------------------------------------------------

    def apply_constraints(self):
        # Input-Calibratoren
        self.cal_t.apply_constraints()
        self.cal_r.apply_constraints()
        self._batched_apply_constraints(self.cal_c)
        self._batched_apply_constraints(self.cal_q)

        # Schicht 1 Lattices
        for lat in self.lattices_A:
            lat.apply_constraints()
        for lat in self.lattices_B:
            lat.apply_constraints()
        for lat in self.lattices_C:
            lat.apply_constraints()

        # Zwischen-Calibratoren
        self._batched_apply_constraints(self.cal_between)

        # Schicht 2 Lattices
        for lat in self.lattices_L2:
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
#   Die U Ensemble-Outputs werden direkt im Anschluss über dim=1 gemittelt
#   (arithmetisches Mittel monotoner Funktionen bleibt monoton), sodass pro
#   Gruppe/Ressource genau EIN Wert entsteht, der alle U Member berücksichtigt
#   — sonst würde Schicht 2 später nur einen einzelnen Member sehen und der
#   Rest wäre ungenutzt.
#
#   Ausgabe pro k: 3 Skalare (nach Ensemble-Mittelung)  ∈ ℝ  (noch nicht in [0,1])
#   Gesamtausgabe Schicht 1: 3 * K Skalare
#   Parameter: 3 * K * lattice_units * 8   (Lattice-Gewichte selbst sind
#     weiterhin lattice_units-fach vorhanden — nur die Ausgabe wird gemittelt)
#
#   Flacher Index eines Outputs (nach Mittelung):
#     Gruppe g ∈ {0=A, 1=B, 2=C}, Ressource k
#     → flat_idx = g * K + k
#
# -----------------------------------------------------------------------------
# SCHICHT 1→2 — Zwischen-Calibratoren  (ein eigener pro (Gruppe, Ressource))
# -----------------------------------------------------------------------------
#
#   Jeder der 3*K (ensemble-gemittelten) Schicht-1-Outputs bekommt einen
#   eigenen INCREASING NumericalCalibrator mit Keypoints in [0,1].
#   Dieser re-normalisiert den Output zurück auf [0,1], damit Schicht 2
#   ihn als gültige Gitterkoordinate verwenden kann.
#
#   Da die Monotonie-Richtung bereits durch die Input-Calibratoren kodiert
#   ist, genügt INCREASING für alle Zwischen-Calibratoren.
#
#   Parameter: 3 * K * P
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
#   Jedes Tripel hat die Form (A_kA, B_kB, C_kC):
#     — immer eine aus jeder Gruppe (A, B, C)
#     — kA, kB, kC bei K>=3 immer drei verschiedene Ressourcen
#       (rotierender Offset i, i+1, i+2 mod K), bei K<3 die maximal
#       mögliche Anzahl
#
#   _build_cross_resource_triplets erzeugt genau K Tripel (n_triplets = K),
#   nicht alle K^3 möglichen (kA,kB,kC)-Kombinationen — sonst würde Schicht 2
#   kubisch statt linear mit der Ressourcenzahl K wachsen (bei K=5 z.B. 125
#   statt 5 Lattices). Da jede (Gruppe, Ressource)-Kombination bereits über
#   alle lattice_units Ensemble-Member gemittelt wurde, deckt jedes Tripel-
#   Element automatisch die volle Schicht-1-Information ab.
#
#   Dadurch kann Schicht 2 lernen, wie z.B. die Kapazitätssituation
#   von Ressource 1 mit dem Zeitdruck-Reward-Profil von Ressource 2
#   interagiert — was Schicht 1 strukturell nicht modellieren kann.
#
#   Alle Lattices in Schicht 2 sind INCREASING (Richtung steckt in den
#   Input-Calibratoren und wird durch INCREASING Zwischen-Calibratoren
#   durchgereicht).
#
#   Ausgabe: K * lattice_units2 Skalare
#   Parameter: K * lattice_units2 * 8
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
#   Parameter: (K * lattice_units2 + 1) * output_dim
#
# -----------------------------------------------------------------------------
# PARAMETERÜBERSICHT (Beispiel K=2, P=8, U=4, U2=2, D_out=2)
# -----------------------------------------------------------------------------
#
#   Input-Calibratoren  (2+2*2)*8         =  48
#   Lattice Schicht 1   3*2*4*8           = 192
#   Zwischen-Calibrat.  3*2*8             =  48
#   Lattice Schicht 2   K=2 Triplets*2*8  =  32
#   Output-Layer        (4+1)*2           =  10
#                                    Σ  = 330
#
# =============================================================================