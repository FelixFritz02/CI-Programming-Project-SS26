# Modellarchitekturen im Ordner `models`

Diese Datei beschreibt die wichtigsten Modelltypen im `Programming Part/models`-Ordner,
ihre Architekturprinzipien sowie typische Stärken und Schwächen.

---

## 1. `standard_dqn.py`

### Architektur
- Einfaches Feedforward-Netzwerk für Q-Wert-Approximation.
- Layer: `input_dim → 256 → ReLU → 256 → ReLU → 128 → ReLU → output_dim`.
- Keine monotone Struktur oder spezialisierte Feature-Transformation.

### Stärken
- Einfach, schnell zu trainieren.
- Gut geeignet als Basis- oder Vergleichsmodell.
- Skalierbar für generische Zustands-Aktions-Räume.

### Schwächen
- Kein strukturelles Wissen über Monotonie oder Domänenbeziehungen.
- Kann mehr Daten benötigen, um strukturierte Relationen zu lernen.
- Weniger interpretierbar als lattice-basierte Modelle.

---

## 2. `lattice_dqn.py`

### Architektur
- Kombiniert ein Standard-DQN mit einem monotonen Lattice-Bonus.
- Pfad A: DQN über alle Features `x = [t, C_1..C_K, r, q_1..q_K]`.
- Pfad B: Nur Kapazitätsfeatures `C_1..C_K` werden durch Kalibratoren und ein K-dimensionales Lattice geführt.
- Finale Q-Werte: `Q = Q_dqn + Q_lattice`.

### Idee
- Das Lattice modelliert einen aktionsunabhängigen Kapazitätsbonus.
- Das DQN lernt aktionsspezifische Unterschiede.

### Stärken
- Nutzt monotone Struktur in Kapazitätsfeatures.
- Hält Aktionsmodellierung flexibel durch das DQN.
- Leichtgewichtigere Integration von Domainwissen.

### Schwächen
- Der Lattice-Bonus ist für alle Aktionen gleich, sodass aktionsspezifische Kapazitätsinteraktionen nicht vollständig erfasst werden.
- Kann bei sehr hoher Kapazitätsdimensionierung schnell komplex werden.

---

## 3. `lattice_dqn_withaction.py`

### Architektur
- Kombination aus Standard-DQN und aktionalem Lattice.
- Pfad A: DQN über alle Features zur Repräsentation von `Q_dqn(s,a)`.
- Pfad B: Monotones Lattice, das `C_1..C_K` plus die Aktion `a` als Input erhält.
- Aktion wird über einen separaten Kalibrator transformiert.
- Finale Q-Werte: `Q(s,a) = Q_dqn(s,a) + Q_lattice(s,a)`.

### Idee
- Das Lattice liefert einen Zusatzwert, der sowohl auf Kapazität als auch auf Aktion basiert.
- Monotonie gilt nur gegenüber den Kapazitätsfeatures, nicht gegenüber der Aktion.

### Stärken
- Erfasst aktionsabhängige Kapazitätsverhältnisse besser als `lattice_dqn.py`.
- Bewahrt strukturierte Kapazitätsmonotonie, erlaubt aber flexible Aktionsentscheidungen.
- Gut, wenn Aktion und Kapazität gemeinsam die Q-Bewertung beeinflussen.

### Schwächen
- Höherer Rechenaufwand, weil das Lattice pro Aktion ausgewertet wird.
- Aktionale Eingabe kann bei vielen Aktionen teuer werden.
- Monotonieannahme ist nur partiell; das Modell ist nicht vollständig monoton in allen Eingaben.

---

## 4. `full_lattice.py`

### Architektur
- Reines Lattice-Netzwerk zur direkten Q-Wert-Approximation.
- Alle Rohfeatures werden zuerst durch monotone `NumericalCalibrator` transformiert.
- Für jede Ressource `k` gibt es drei 3-Input-Lattice-Gruppen:
  - Gruppe A: `[t, r, C_k]`
  - Gruppe B: `[C_k, q_k, r]`
  - Gruppe C: `[t, C_k, q_k]`
- Ausgaben aller Lattices werden durch einen Linear-Layer mit nicht-negativen Gewichten zu `Q(s,a)` kombiniert.

### Idee
- Reine monotone Lattice-Struktur mit domänenspezifischen Monotonieannahmen.
- Unterstützt interpretierbare, strukturierte Beziehungen.

### Stärken
- Vollständig monotone Architektur in den definierten Feature-Richtungen.
- Interpretierbare Feature-Kombinationen und klare Domänenstruktur.
- Keine Black-Box-MLP-Komponente im Hauptpfad.

### Schwächen
- Kann bei größerer Ressourcenanzahl schnell parametrisch wachsen.
- Eingeschränkte Flexibilität gegenüber nicht-monotonen Beziehungen.
- Relativ komplexe Architektur, die mehr Zeit für Tuning erfordern kann.

---

## 5. `deep_lattice.py`

### Architektur
- Zweistufiges Lattice-Netzwerk mit zusätzlicher Zwischenkalibrierung.
- Schicht 0: Input-Calibratoren für `t, C_k, r, q_k`.
- Schicht 1: Ensemble von 3-Input-Lattices in Gruppen A/B/C pro Ressource.
- Zwischenkalibratoren normalisieren Schicht-1-Ausgaben auf `[0,1]`.
- Schicht 2: Cross-Resource-Lattices, die kombinierte Ausgaben verschiedener Gruppen und Ressourcen lernen.
- Output-Layer mit nicht-negativen Gewichten aggregiert zu `Q(s,a)`.

### Idee
- Erweitert das reine Lattice-Modell um cross-resource Interaktionen.
- Stellt sicher, dass die Monotonie durch alle Schichten erhalten bleibt.

### Stärken
- Stärkere Modellierung komplexer Interaktionen zwischen Ressourcen.
- Geeignet für Situationen, in denen mehrere Ressourcen und Features sich gegenseitig beeinflussen.
- Erhöhte Ausdruckskraft bei Beibehaltung monotone Struktur.

### Schwächen
- Noch größerer Rechen- und Parameteraufwand als `full_lattice.py`.
- Höhere Architekturkomplexität erschwert Debugging und Hyperparameter-Tuning.
- Potenzielle Überanpassung bei kleinen Datensätzen.

---

## Vergleich und Empfehlung

- `standard_dqn.py`: Beste Wahl als einfache Baseline oder wenn keine monotone Struktur bekannt ist.
- `lattice_dqn.py`: Gut, wenn ein globaler kapazitätsabhängiger Bias plus DQN ausreicht.
- `lattice_dqn_withaction.py`: Besser, wenn Aktion und Kapazität gemeinsam wichtige Q-Effekte haben.
- `full_lattice.py`: Geeignet bei klaren monotone Beziehungen und einem Bedarf an interpretierbarer Architektur.
- `deep_lattice.py`: Am besten, wenn aufwändigere Interaktionen zwischen Ressourcen modelliert werden sollen und genug Daten vorhanden sind.
