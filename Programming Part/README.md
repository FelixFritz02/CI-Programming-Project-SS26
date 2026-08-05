## Breaking Changes

- **neue** Ordner-Struktur

- Notebooks haben einheitliche Struktur

- Frühabbruch wenn Epsiode nur noch Ablehnen würde in `DrauspEnv.step()`: Existiert nach einem Schritt keine
  zusammenhängende Kapazitätslücke mehr, in die eine Anfrage reinpasst, wird die Episode sofort beendet.
  Bisher ist man bis `T_d` nur noch mit "Ablehnen"-Schritten weitergelaufen. Reduziert nutzlose Replay-Buffer-Einträge. Greift nur, wenn `fixed_request_length` gesetzt ist.

- **`DQNAgent`: `epsilon_decay` (episodenbasiert) ersetzt durch `epsilon_decay_steps`
  (schrittbasiert).** Epsilon fällt jetzt exponentiell über eine feste Anzahl an
  *Umgebungsschritten* ab statt pro Episode multiplikativ. Grund: seit Episoden durch die
  Frühabbruch-Logik (siehe oben) unterschiedlich lang sind, ließ episodenbasiertes Decay die
  Exploration inkonsistent schnell auslaufen.

- **Neuer Parameter `reject_bias` in `DQNAgent`:** während der
  Epsilon-Greedy-Exploration wird mit dieser Wahrscheinlichkeit aktiv abgelehnt (Aktion 0)
  statt uniform zufällig unter allen gültigen Aktionen zu wählen. 
  Ziel: zufällige Exploration kommt tiefer in die Instanz

- **`DQNAgent.train()` gibt jetzt 3 statt 2 Werte zurück:**
  `reward_history, monotonicity_history, depth_history`. `depth_history` enthält pro Episode
  `(episode, erreichte_tiefe, epsilon)` — wie viele Umgebungsschritte die Episode überlebt hat,
  bevor sie terminiert wurde. Neuer Plot!
    --> Man sieht ob es der Agent schafft die gesamte Instanz zu sehen

- Gamma (Diskontierungsfaktor) erhöht --> läuft damit viel besser (Zukunft spielt größere Rolle --> späteres Annehmen, tief in der Instanz, wird wichtiger)

- **`T_d` und Pool-Größe `|N|` waren verwechselt — jetzt sauber getrennt.** Beim Laden einer
  Benchmark-Instanz (`instance_reader.get_instance_data`) ist `instance` jetzt ein **Pool** von
  `|N|` Kandidaten-Anfragen (Feld umbenannt: `DrauspInstanceData.num_requests` →
  `DrauspInstanceData.pool_size`), aus dem `DrauspEnv.reset()` pro Episode `T_d` Anfragen
  **zufällig mit Zurücklegen** zieht — wie im Paper (`drausp_lion18.pdf`) und in der gegebenen
  Referenzimplementierung (`ci_project_kloster/src/drausp_env.py`) beschrieben, statt die Datei
  einmal deterministisch von vorne durchzulaufen. `T_d` ist jetzt ein frei wählbarer Parameter
  (vorher fälschlich = Zeilenzahl der Datei) und muss beim Environment-Aufbau explizit gesetzt
  werden (siehe Notebooks: `T_d = 10` mit Kommentar, welche Paper-Instanz das entspricht). Auch
  `training/monotonicity_evaluation.py` testet jetzt gegen die zuletzt gezogene
  Episoden-Sequenz (`env._instance`) statt gegen den rohen Pool (`env._fixed_instance`) — behebt
  nebenbei einen Crash beim Training ohne festen Pool.

## Offene Baustellen

- Kloster nach Benchmarks für Instanzen fragen --> sind im Paper!!
- Monotonie-Test überprüfen, testet aktuell nur mit deterministischen C_k-Werten
- Monotonie-Fehlerterm:
    - dynamisches Mono-Lambda setzen
    - Erweiterung auf t,q und r (Fehlerterm wird bislang nur bei Verletzung der Monotonie in C_k ergänzt)
- Full-Lattice Netzwerk weiterentwickeln

- auf verschiedenen Instanzen trainieren und auf ungesehenen Instanzen (mit gleicher Requestgröße testen)
    --> Idee: vielleicht bringt hier die Monotonie einen Vorteil (robusteres / generalisiertes Modell)
- Bericht anfangen
