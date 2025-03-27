import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
import os
import random

#MODEL_PATH = "./saved_models/dicewars_rl_model.h5"
MODEL_PATH = "./saved_models/dicewars_rl_model_new.keras"

class RLDicewarsAgent:
    def __init__(self):
        print("Inizializzazione agente RL")
        self.input_dim = 60  # <-- n* di aree sullla mappa
        self.output_dim = 50  # <-- scegli in base a quante azioni possibili consideri
        #self.output_dim = len(all_actions)

        self.build_action_mapping()  # inizializza il mapping
        #self.output_dim = len(self.index_to_action)  # <-- imposta in base alle azioni reali
        self.model = self.build_model()
        self.load_model()
        
    def build_model(self):
        with tf.device('/GPU:0'):  ### aa commentare se non abbiamo gpu
            model = keras.Sequential([
                        layers.Input(shape=(self.input_dim,)),         # es: 60
                        layers.Dense(256, activation='relu'),
                        layers.BatchNormalization(),                   # migliora la stabilità
                        layers.Dense(128, activation='relu'),
                        layers.Dropout(0.2),                           # evita overfitting su dati poveri
                        layers.Dense(64, activation='relu'),
                        layers.Dense(self.output_dim, activation='linear')
                    ])
            model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=MeanSquaredError())
        return model
    
    def save_model(self, path=MODEL_PATH):
        self.model.save(path)
        print(f"[MODEL] Salvato in: {path}")

    def load_model(self, path=MODEL_PATH):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            print(f"[MODEL] Caricato da: {path}")
        else:
            print(f"[MODEL] Nessun file trovato in: {path}, si parte da zero.")

    def encode_state(self, grid, match_state):
        """
        Converti grid e match_state in un vettore numerico di dimensione `input_dim`
        """
        flat_dice = np.array(match_state.area_num_dice) / 8.0
        flat_owner = np.array(match_state.area_players) / 3.0
        state_vec = np.concatenate([flat_dice, flat_owner])
        ##print("Input shape:", state_vec.shape)
        # Padding o taglio
        target_len = self.input_dim
        if len(state_vec) < target_len:
            padding = np.zeros(target_len - len(state_vec))
            state_vec = np.concatenate([state_vec, padding])
        else:
            state_vec = state_vec[:target_len]
        
        return state_vec

    def decode_action(self, action_idx, grid, match_state):
        """
        Mappa l'indice in una coppia (from_area, to_area) oppure None
        """
        valid_actions = self.get_valid_actions(grid, match_state)
        if action_idx < len(valid_actions):
            return valid_actions[action_idx]
        return None

    def build_action_mapping(self):
        """
        Costruisce un mapping fisso per tutte le azioni possibili.
        In questo esempio, assumiamo che il numero massimo di aree sia fisso (es. 30).
        Puoi adattare la logica in base alla tua implementazione.
        """
        #num_areas = len(grid.areas)
        #print("Numero reale di aree:", num_areas)

        max_areas = 30  # modifica questo valore in base alla tua griglia reale
        all_actions = []
        for from_area in range(max_areas):
            for to_area in range(max_areas):
                if from_area != to_area:
                    all_actions.append((from_area, to_area))
        self.action_to_index_dict = {action: idx for idx, action in enumerate(all_actions)}
        self.index_to_action_dict = {idx: action for idx, action in enumerate(all_actions)}
        self.output_dim = len(all_actions)

    def get_valid_actions(self, grid, state):
        # Ottieni le aree possedute dall'agente (player 0)
        player_areas = state.player_areas[0]
        actions = []
        for from_area in player_areas:
            # Per ogni area posseduta, ottieni i vicini usando grid.areas
            for to_area in grid.areas[from_area].neighbors:
                # Verifica se il vicinante non è posseduto dall'agente (player 0)
                if state.area_players[to_area] != 0:  # usa state.area_players per conoscere i proprietari
                    action = (from_area, to_area)
                    if action in self.action_to_index_dict:
                        idx = self.action_to_index_dict[action]
                        actions.append(idx)

        # stampa per debug
        #print("Azioni valide:", [self.index_to_action_dict[i] for i in actions])
        return actions


    def select_action(self, grid, match_state, epsilon=0.1):
        state_vec = self.encode_state(grid, match_state)
        valid_indices = self.get_valid_actions(grid, match_state)

        if not valid_indices:
            return None  # nessuna azione valida

        if np.random.rand() < epsilon:
            chosen_idx = np.random.choice(valid_indices)
        else:
            q_values = self.model.predict(state_vec[None, :], verbose=0)[0]
            sub_q = [q_values[i] for i in valid_indices]
            chosen_idx = valid_indices[np.argmax(sub_q)]

        return self.index_to_action_dict[chosen_idx]  # ritorna (from, to)


    
    def train_step(self, state_vec, action_taken, reward, gamma=0.99):
        state_vec = state_vec[None, :]  # batch format
        q_values = self.model.predict(state_vec, verbose=0)
        target_q = q_values.copy()

        valid_actions = self.get_valid_actions_dummy()  # placeholder per max output
        action_idx = self.action_to_index(action_taken, valid_actions)

        target_q[0, action_idx] = reward  # valore target semplice (può diventare più sofisticato)
        self.model.fit(state_vec, target_q, verbose=0)

    def get_valid_actions_dummy(self):
        """Nota: get_valid_actions_dummy è un trucco per allenamento offline.
        Durante il training reale potresti voler passare anche grid, state per creare un mapping valido ogni volta."""
        # Dummy: assume output dim costante (50) per compatibilità
        return [i for i in range(self.output_dim)]

    def action_to_index(self, action, valid_actions):
        if action is None:
            return 0
        try:
            return valid_actions.index(action)
        except ValueError:
            return 0
        
    def train_batch(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = np.array(states)
        next_states = np.array(next_states)

        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        targets = q_values.copy()

        valid_actions = self.get_valid_actions_dummy()

        for i, (a, r, done) in enumerate(zip(actions, rewards, dones)):
            ##idx = self.action_to_index(a)
            idx = self.action_to_index(a, valid_actions)
            if idx is None:
                continue  # salta se non trova l'indice (safety)
            target = r if done else r + gamma * np.max(q_next[i])
            ###print(f"Stato {i}: Q-value previsto: {q_values[i][idx]}, Target: {target}")

            targets[i, idx] = target

        self.model.fit(states, targets, verbose=0)
        
    def generate_action_mappings(self):
        mapping = {}
        reverse = {}
        idx = 0
        max_areas = 30  # usa il numero massimo di aree sulla mappa

        for from_area in range(max_areas):
            for to_area in range(max_areas):
                if from_area != to_area:
                    action = (from_area, to_area)
                    mapping[idx] = action
                    reverse[action] = idx
                    idx += 1

        return mapping, reverse


### ==================

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), list(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
