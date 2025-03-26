# machine-learning

Algoritmi: Q-learning, Deep Q-Network (DQN), Policy Gradient, ecc.

Dal PDF del progetto *Machine Learning 5EC - Dicewars* e dai file allegati, ecco gli **step principali** che dovresti seguire con **focus sulla creazione dell’IA con Reinforcement Learning (RL)**:

---

### 🔁 **Panoramica Generale del Progetto**
L'obiettivo è sviluppare un **agente intelligente (Player)** in grado di **giocare e vincere a Dicewars**, usando un approccio di **Reinforcement Learning** (nessun dataset, solo esperienza tramite simulazione delle partite).

#### ✅ 1. **Studia il Reinforcement Learning**
- Comprendere concetti chiave:
  - Agente, ambiente, stato, azione, reward.
  - Algoritmi: Q-learning, Deep Q-Network (DQN), Policy Gradient, ecc.
- Usa come riferimento il **notebook didattico RL**, articoli o video YouTube.

---

#### 🧠 2. **Progetta il tuo Agente RL**
- Definisci cosa sarà:
  - **Stato** = `grid` + `state`: informazioni sul gioco in corso.
  - **Azione** = coppia (from_area, to_area) o `None` (fine turno).
  - **Reward** = lo definisci tu, ad esempio:
    - +1 per una conquista di territorio
    - +5 per una vittoria
    - -1 per perdita di territorio
    - 0 altrimenti

> ✍️ **Suggerimento**: Inizia con un modello semplice e poi aumenta la complessità.

---

#### 🛠️ 3. **Implementa il tuo Agente RL**
All’interno di `playergroupX.py`:
- Sottoclasse della classe `Player`
- Implementa `get_attack_areas(self, grid, match_state)` per restituire l’azione da compiere
- Usa TensorFlow/Keras per costruire il modello (`self.model = keras.Sequential(...)`)

> 📌 Puoi inizialmente collezionare dati da partite casuali (`RandomPlayer`) per fare esperienza offline prima di passare a un training online.

---

#### 🧪 4. **Crea l’ambiente di training**
- Usa `basic_dicewars.py` o `dicewars_contest.py` come loop di simulazione.
- Per ogni turno:
  1. Osserva `state`
  2. Seleziona azione (`get_attack_areas`)
  3. Passa l’azione a `match.step(action)`
  4. Ottieni nuovo stato e valuta il reward
  5. Allenati sul reward

> 🔁 Ripeti per molte partite (es. 1000+) per migliorare la strategia del tuo agente.

---

#### ⚙️ 5. **Ottimizza e Valida**
- Allena e registra metriche:
  - Percentuale di vittorie
  - Progressione del reward
- Modifica iperparametri (learning rate, gamma, ecc.)
- Testa contro altri agenti: `AgressivePlayer`, `DefaultPlayer`, ecc.

---

### 📝 Deliverables
1. ✅ **Un agente AI funzionante** che giochi tramite `get_attack_areas(grid, state)`
2. 📊 **Report PDF** con:
   - Metodo RL usato
   - Architettura del codice
   - Strategia di reward
   - Grafico delle performance
   - Conclusioni, limiti e miglioramenti futuri

---

### 📅 Deadline
- **Consegna modello per il contest**: *7 Aprile 2025 ore 22:00*
- **Consegna report finale**: *17 Aprile 2025 ore 22:00*

---

### 💡 Consiglio Finale
Puoi iniziare con un modello tipo Deep Q-Network (DQN) o anche Q-table se semplifichi lo stato. L’importante è impostare bene il ciclo osservazione → azione → reward → apprendimento.

---

Fammi sapere se vuoi un esempio base di agente RL (DQN) o uno script per raccolta dati / training.