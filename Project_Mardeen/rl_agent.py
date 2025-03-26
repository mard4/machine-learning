import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

MODEL_PATH = "saved_models/dicewars_rl_model.h5"

class RLDicewarsAgent:
    def __init__(self):
        self.input_dim = 60  # <-- n* di aree sullla mappa
        self.output_dim = 50  # <-- scegli in base a quante azioni possibili consideri
        self.model = self.build_model()
        self.load_model()
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.output_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
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

    def get_valid_actions(self, grid, match_state):
        """
        Restituisce tutte le azioni possibili del tipo (from_area, to_area) + [None]
        """
        from_player = match_state.player
        player_areas = match_state.player_areas
        area_num_dice = match_state.area_num_dice
        actions = [None]

        for from_area in player_areas[from_player]:
            if area_num_dice[from_area] > 1:
                for to_area in grid.areas[from_area].neighbors:
                    if to_area not in player_areas[from_player]:
                        actions.append((from_area, to_area))
        return actions

    def select_action(self, grid, match_state):
        """
        Metodo chiamato dal player esterno per scegliere la mossa
        """
        state_vec = self.encode_state(grid, match_state)
        q_values = self.model.predict(state_vec[None, :], verbose=0)[0]
        valid_actions = self.get_valid_actions(grid, match_state)

        # Scegli la migliore azione valida
        action_scores = q_values[:len(valid_actions)]
        best_idx = np.argmax(action_scores)
        return valid_actions[best_idx]
    
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

