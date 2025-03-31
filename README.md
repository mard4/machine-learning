# machine-learning

Algoritmi: Q-learning, Deep Q-Network (DQN), PPO, ecc.

NOTA: Ricordarsi di modificare i path di salvataggio del modello in training ed nel codice agent.

### 🔁 **Panoramica Generale del Progetto**
L'obiettivo è sviluppare un **agente intelligente (Player)** in grado di **giocare e vincere a Dicewars**, usando un approccio di **Reinforcement Learning** (nessun dataset, solo esperienza tramite simulazione delle partite).

#### 🧠 **Agente RL**
  - **Stato** = `grid` + `state`: informazioni sul gioco in corso.
  - **Azione** = coppia (from_area, to_area) o `None` (fine turno).
  - **Reward** = lo definisci tu, ad esempio:
    - +1 per una conquista di territorio
    - +5 per una vittoria
    - -1 per perdita di territorio
    - 0 altrimenti
   
  in `rl_agent.py`

#### 🧪 **Crea l’ambiente di training**
- `basic_dicewars.py` o `dicewars_contest.py` come loop di simulazione.
- Per ogni turno:
  1. Osserva `state`
  2. Seleziona azione (`get_attack_areas`)
  3. Passa l’azione a `match.step(action)`
  4. Ottieni nuovo stato e valuta il reward
  5. Allenati sul reward

> 🔁 Ripeti per molte partite (es. 1000+) per migliorare la strategia del tuo agente.

in `rl_training.py`

---

#### ⚙️ **Ottimizza e Valida**
- Allena e registra metriche:
  - Percentuale di vittorie
  - Progressione del reward
- Modifica iperparametri (learning rate, gamma, ecc.)
- Testa contro altri agenti: `AgressivePlayer`, `DefaultPlayer`, ecc. (o anche contro se stesso?)
