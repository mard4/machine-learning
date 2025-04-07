import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from dicewars.grid import Grid
import random
import os
from collections import Counter

MODEL_PATH = "D:/PhD utwente/courses/Machine learning/Exercises/dicewars-env-v1/saved_models/dicewars_rl_model.keras"

class RLDicewarsAgent:
    def __init__(self):
        print("Initialized RL agent")
        self.grid = Grid()
        self.input_dim = 21  
        self.output_dim = 57 #393  
        self.model = self.build_model()
        #self.load_model()
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),        
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.output_dim, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss=MeanSquaredError())
        #model.compile(optimizer=optimizers.Adam(learning_rate=2e-4), loss='huber')
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
        
        dice_per_area = np.array(match_state.area_num_dice)
        owners = match_state.area_players
        max_dice_counts = np.zeros(len(match_state.player_num_dice))
        min_dice_counts = np.zeros(len(match_state.player_num_dice))

        for area_idx, owner in enumerate(owners):
            dice_count = dice_per_area[area_idx]
            if dice_count == 8:
                max_dice_counts[owner] += 1
            elif dice_count == 1:
                min_dice_counts[owner] += 1

        max_dice_counts = max_dice_counts / Grid.DEFAULT_MAX_NUM_AREAS
        min_dice_counts = min_dice_counts / Grid.DEFAULT_MAX_NUM_AREAS

        # Ordina anche questi vettori con il tuo player in testa
        my_max = max_dice_counts[match_state.player]
        other_max = np.delete(max_dice_counts, match_state.player)
        other_max = np.sort(other_max)
        max_dice_vec = np.concatenate(([my_max], other_max))

        my_min = min_dice_counts[match_state.player]
        other_min = np.delete(min_dice_counts, match_state.player)
        other_min = np.sort(other_min)
        min_dice_vec = np.concatenate(([my_min], other_min))
        
        # stock = np.array(match_state.player_num_stock)/8.0
        # my_stock = stock[match_state.player]
        # other_stock = np.delete(stock, match_state.player)
        # other_stock = np.sort(other_stock)
        # stock = np.concatenate(([my_stock], other_stock))
        
        enemy_borders = 0
        
        for area in match_state.player_areas[match_state.player]:
            enemy_borders += sum(1 for n in grid.areas[area].neighbors if match_state.area_players[n] != match_state.player)
        
        enemy_borders = np.array([enemy_borders / Grid.DEFAULT_MAX_NUM_AREAS])

        state_vec = np.concatenate([dice, areas, clusters, max_dice_vec, min_dice_vec, enemy_borders])
        return state_vec
    

    def decode_action(self, action_idx, grid, match_state):
        
        valid_actions, _ = self.get_valid_actions(grid, match_state)
        dice = (action_idx // 8 + 2, action_idx % 8 + 1)
        np.random.shuffle(valid_actions[1:])
        if action_idx == 56:
            return None
        for action in valid_actions[1:]:
            from_area, to_area = action
            if (match_state.area_num_dice[from_area] == dice[0] and match_state.area_num_dice[to_area] == dice[1]):
                return action
    
    # def decode_action(self, action_idx, grid, match_state):
    #     if action_idx == 392:
    #         return None

    #     die1_idx = action_idx // (8 * 7)
    #     rem1 = action_idx % (8 * 7)

    #     die2_idx = rem1 // 7
    #     from_neighbors_idx = rem1 % 7

    #     die1 = die1_idx + 2
    #     die2 = die2_idx + 1
    #     from_neighbors = from_neighbors_idx + 1

    #     valid_actions, _ = self.get_valid_actions(grid, match_state)

    #     for action in valid_actions[1:]:
    #         from_area, to_area = action
    #         if (
    #             match_state.area_num_dice[from_area] == die1 and
    #             match_state.area_num_dice[to_area] == die2 and
    #             len(grid.areas[from_area].neighbors) == from_neighbors
    #         ):
    #             return action

    #     return None

            
    def encode_action(self, action, match_state, grid):
        # Caso speciale: nessuna azione
        if action is None:
            return 56

        from_area, to_area = action
        die1 = match_state.area_num_dice[from_area]  # valore tra 2 e 8
        die2 = match_state.area_num_dice[to_area]    # valore tra 1 e 8

        # Calcola indice: (die1 - 2) * 8 + (die2 - 1)
        action_idx = (die1 - 2) * 8 + (die2 - 1)
        return action_idx
    
    # def encode_action(self, action, match_state, grid):
    #     if action is None:
    #         return 392  # ultima uscita = None

    #     from_area, to_area = action
    #     die1 = match_state.area_num_dice[from_area]
    #     die2 = match_state.area_num_dice[to_area]
    #     from_neighbors = len(grid.areas[from_area].neighbors)

    #     idx = (
    #         (die1 - 2) * 8 * 7 +
    #         (die2 - 1) * 7 +
    #         (from_neighbors - 1)
    #     )
    #     return idx

    
    
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
                        
        # from_area_counts = Counter(from_area for from_area, _ in actions[1:])
        # to_area_counts = Counter(to_area for _ , to_area in actions[1:])
                        
        # def sort_key(action):
        #     from_area, to_area = action
            
        #     # opponent_neighbor_count = 0
        #     # opponent_dice = 0
        #     # for neighbor in grid.areas[from_area].neighbors:
        #     #     if match_state.area_players[neighbor] != match_state.player:  # Se il vicino è di un avversario
        #     #         opponent_neighbor_count += 1
        #     #         opponent_dice += match_state.area_num_dice[neighbor]
                
        #     return (len(grid.areas[from_area].neighbors), area_num_dice[from_area], area_num_dice[to_area], len(grid.areas[to_area].neighbors))
        #     # return (from_area_counts[from_area], area_num_dice[from_area], to_area_counts[to_area], area_num_dice[to_area])
        
        # valid_actions_sorted = sorted(actions[1:], key=sort_key, reverse=True)
        # return valid_actions_sorted
        
        possible_idx = []
        for a in actions:
            possible_idx.append(self.encode_action(a, match_state, grid))
        
        return actions, possible_idx

    def select_action(self, grid, match_state, epsilon=0.1):
        state_vec = self.encode_state(grid, match_state)
        valid_actions, possible_idx = self.get_valid_actions(grid, match_state)
        #print(valid_actions, possible_idx)

        if not valid_actions:
            return 0, None  

        if np.random.rand() < epsilon:
            action = random.choice(valid_actions)
            #action_idx = valid_actions.index(action)
            action_idx = self.encode_action(action, match_state, grid)
            return action_idx, action
        else:
            q_values = self.model.predict(state_vec[None, :], verbose=0)[0]
            #action_indices = [hash(action) % self.output_dim for action in valid_actions]
            best_action_idx = max(possible_idx, key=lambda i: q_values[i])
            #print(best_action_idx, self.decode_action(best_action_idx, grid, match_state))
            #return best_action_idx, valid_actions[best_action_idx]
            return best_action_idx, self.decode_action(best_action_idx, grid, match_state)

    def train_batch(self, states, actions_indices, actions, rewards, next_states, dones, gamma=0.9, alpha=0.5):
        states = np.array(states)
        next_states = np.array(next_states)

        q_values = self.model.predict(states, verbose=0)
        #print(q_values.shape)
        q_next = self.model.predict(next_states, verbose=0)
        targets = q_values.copy()

        for i, (action_idx, reward, done) in enumerate(zip(actions_indices, rewards, dones)):
            if done:
                targets[i, action_idx] = (1-alpha)*q_values[i, action_idx] + alpha*reward
            else:
                targets[i, action_idx] = (1-alpha)*q_values[i, action_idx] + alpha*(reward + gamma * np.max(q_next[i]))
                


        self.model.fit(states, targets, verbose=0)


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action_idx, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action_idx, action, reward, next_state, done))

    def sample(self, batch_size):
        
        # high_reward_idxs = [i for i, experience in enumerate(self.buffer) if experience[3] >= 20]
        # if int(batch_size-len(high_reward_idxs)) > 0:
        #     idxs = np.random.choice(len(self.buffer), size= int(batch_size-len(high_reward_idxs)), replace=False)
        # else:
        #     idxs = np.array([], dtype=int)
        
        high_reward = [exp for exp in self.buffer if exp[3] > 10.0]
        random_batch = random.sample(self.buffer, batch_size - len(high_reward))
        batch = high_reward + random_batch
        
        # idxs = np.random.choice(len(self.buffer), size= batch_size, replace=False)
        
        # #selected_idxs = np.concatenate([high_reward_idxs, idxs])
        # batch = [self.buffer[i] for i in idxs]
        states, actions_idx, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions_idx), list(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
