import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from dicewars.grid import Grid
import random
import os

MODEL_PATH = "D:/PhD utwente/courses/Machine learning/Exercises/dicewars-env-v1/saved_models/dicewars_rl_model.keras"

class RLDicewarsAgent:
    def __init__(self):
        print("Initialized RL agent")
        self.grid = Grid()
        self.input_dim = 12  
        self.output_dim = 80  
        #self.model = self.build_model()
        self.load_model()
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),        
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
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
            
    
    def encode_state(self, grid, match_state):  # da rivedere, usare descrittori
        dice = np.array(match_state.player_num_dice)/8.0
        my_dice = dice[match_state.player]
        other_dice = np.delete(dice, match_state.player)
        other_dice = np.sort(other_dice)
        dice = np.concatenate(([my_dice], other_dice))
        
        areas = np.array(match_state.player_num_areas)/Grid.DEFAULT_MAX_NUM_AREAS
        my_areas =  areas[match_state.player]
        other_areas = np.delete(areas, match_state.player)
        other_areas = np.sort(other_areas)
        areas = np.concatenate(([my_areas], other_areas))
        
        clusters = np.array(match_state.player_max_size)/Grid.DEFAULT_MAX_NUM_AREAS
        my_clusters =  clusters[match_state.player]
        other_clusters = np.delete(clusters, match_state.player)
        other_clusters = np.sort(other_clusters)
        clusters = np.concatenate(([my_clusters], other_clusters))
           
        state_vec = np.concatenate([dice, areas, clusters])
        
        ##print("Input shape:", state_vec.shape)
        # target_len = self.input_dim
        # if len(state_vec) < target_len:
        #     padding = np.zeros(target_len - len(state_vec))
        #     state_vec = np.concatenate([state_vec, padding])
        # else:
        #     state_vec = state_vec[:target_len]
        
        return state_vec
    

    def decode_action(self, action_idx, grid, match_state):
        
        valid_actions = self.get_valid_actions(grid, match_state)
        if action_idx < len(valid_actions):
            return valid_actions[action_idx]
        return None
    
    
    def get_valid_actions(self, grid, match_state):
        from_player = match_state.player
        player_areas = match_state.player_areas
        area_num_dice = match_state.area_num_dice
        actions = [None]

        for from_area in player_areas[from_player]:
            if area_num_dice[from_area] > 1:
                for to_area in grid.areas[from_area].neighbors:
                    if to_area not in player_areas[from_player]:
                        actions.append((from_area, to_area))
                        
        def sort_key(action):
            from_area, to_area = action
            # Restituisci una tupla con il numero di dadi nelle due aree
            return (len(grid.areas[from_area].neighbors), len(grid.areas[to_area].neighbors), area_num_dice[from_area], area_num_dice[to_area])
        
        valid_actions_sorted = sorted(actions[1:], key=sort_key, reverse=True)
        return valid_actions_sorted

    def select_action(self, grid, match_state, epsilon=0.1):
        state_vec = self.encode_state(grid, match_state)
        valid_actions = self.get_valid_actions(grid, match_state)

        if not valid_actions:
            return None  

        if np.random.rand() < epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.model.predict(state_vec[None, :], verbose=0)[0]
            #print(q_values.shape)
            #action_indices = [hash(action) % self.output_dim for action in valid_actions]
            best_action_idx = np.argmax([q_values[i] for i in range(min(self.output_dim, len(valid_actions)))])
            return valid_actions[best_action_idx]

    def train_batch(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = np.array(states)
        next_states = np.array(next_states)

        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        targets = q_values.copy()

        for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
            action_idx = hash(action) % self.output_dim
            if done:
                targets[i, action_idx] = reward
            else:
                targets[i, action_idx] = reward + gamma * np.max(q_next[i])

        self.model.fit(states, targets, verbose=0)


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
