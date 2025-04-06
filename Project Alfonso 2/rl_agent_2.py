import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from dicewars.grid import Grid
import random
import os
from collections import Counter

MODEL_PATH = os.path.join("saved_models", "dicewars_rl_model_vsrandom_3.keras")

class RLDicewarsAgent:
    def __init__(self):
        print("Initialized RL agent")
        self.grid = Grid()
        self.input_dim = 23  
        self.output_dim = 57  
        #self.model = self.build_model()
        self.load_model()
        
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),        
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.output_dim, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-9), loss=MeanSquaredError())
        return model
    
    def save_model(self, path=MODEL_PATH):
        self.model.save(path)
        print(f"[MODEL] Salvato in: {path}")

    def load_model(self, path=MODEL_PATH):
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path)
                print(f"[MODEL] Caricato da: {path}")
            except Exception as e:
                print(f"[MODEL] ERRORE nel caricamento, ricreo modello da zero. Errore: {e}")
                self.model = self.build_model()
        else:
            print(f"[MODEL] Nessun file trovato in: {path}, creo nuovo modello.")
            self.model = self.build_model()
            
    
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
        
        enemy_borders = 0
        
        for area in match_state.player_areas[match_state.player]:
            enemy_borders += sum(1 for n in grid.areas[area].neighbors if match_state.area_players[n] != match_state.player)
        
        enemy_borders = np.array([enemy_borders / Grid.DEFAULT_MAX_NUM_AREAS])

        
        # descrittore
        b_strength = self.border_strength_ratio(grid, match_state)
        adv = self.dice_advantage(match_state)

        
        ##state_vec = np.concatenate([dice, areas, clusters, max_dice_vec, min_dice_vec, enemy_borders])
        state_vec = np.concatenate([
            dice,
            areas,
            clusters,
            max_dice_vec,
            min_dice_vec,
            enemy_borders,
            [b_strength],
            [adv]
        ])
        return state_vec
    

    def decode_action(self, action_idx, grid, match_state):
        
        valid_actions, _ = self.get_valid_actions(grid, match_state)
        dice = (action_idx // 8 + 2, action_idx % 8 + 1)
        if action_idx == 56:
            return None
        for action in valid_actions[1:]:
            from_area, to_area = action
            if (match_state.area_num_dice[from_area] == dice[0] and match_state.area_num_dice[to_area] == dice[1]):
                return action
            
    def encode_action(self, action, match_state):
        # Caso speciale: nessuna azione
        if action is None:
            return 56

        from_area, to_area = action
        die1 = match_state.area_num_dice[from_area]  # valore tra 2 e 8
        die2 = match_state.area_num_dice[to_area]    # valore tra 1 e 8

        # Calcola indice: (die1 - 2) * 8 + (die2 - 1)
        action_idx = (die1 - 2) * 8 + (die2 - 1)
        return action_idx
    
    
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
            possible_idx.append(self.encode_action(a, match_state))
        
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
            action_idx = self.encode_action(action, match_state)
            return action_idx, action
        else:
            q_values = self.model.predict(state_vec[None, :], verbose=0)[0]
            #action_indices = [hash(action) % self.output_dim for action in valid_actions]
            best_action_idx = max(possible_idx, key=lambda i: q_values[i])
            #print(best_action_idx, self.decode_action(best_action_idx, grid, match_state))
            #return best_action_idx, valid_actions[best_action_idx]
            return best_action_idx, self.decode_action(best_action_idx, grid, match_state)

    def train_batch(self, states, actions_indices, actions, rewards, next_states, dones, gamma=0.9, alpha=0.2):
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
        
        
    #### ==============================
    ### Descrittori
    def border_strength_ratio(self, grid, match_state):
        # Recuperi l'indice del giocatore che sta per agire
        player_idx = match_state.player
        
        total = 0
        count = 0
        area_dice = match_state.area_num_dice
        player_areas = match_state.player_areas[player_idx]
        owner = match_state.area_players

        for from_area in player_areas:
            from_dice = area_dice[from_area]
            for to_area in grid.areas[from_area].neighbors:
                if owner[to_area] != player_idx:
                    to_dice = area_dice[to_area]
                    total += (from_dice - to_dice)
                    count += 1

        return total / count if count > 0 else 0
    
    def dice_advantage(self, match_state):
        """
        Calcola un indicatore che misura quanto sei in vantaggio
        (o in svantaggio) rispetto alla media dei dadi degli avversari.
        Restituisce un singolo float normalizzato.
        """
        my_player = match_state.player
        my_dice = match_state.player_num_dice[my_player]
        total_dice = sum(match_state.player_num_dice)
        others_total = total_dice - my_dice
        others_count = len(match_state.player_num_dice) - 1
        
        if others_count > 0:
            others_avg = others_total / others_count
        else:
            # Caso limite, se fossi l'unico giocatore (poco probabile)
            others_avg = my_dice

        # Normalizziamo la differenza su un fattore empirico (es: 15)
        advantage = (my_dice - others_avg) / 15.0
        return advantage



        # def dice_entropy(self, match_state, player_idx):
        #     from scipy.stats import entropy
        #     dice = np.array([match_state.area_num_dice[i] for i in match_state.player_areas[player_idx]])
        #     if dice.sum() == 0:
        #         return 0
        #     probs = dice / dice.sum()
        #     return entropy(probs)

        # def attack_options(self, grid, match_state, player_idx):
        #     count = 0
        #     player_areas = match_state.player_areas[player_idx]
        #     area_dice = match_state.area_num_dice
        #     owner = match_state.area_players

        #     for from_area in player_areas:
        #         if area_dice[from_area] > 1:
        #             for to_area in grid.areas[from_area].neighbors:
        #                 if owner[to_area] != player_idx:
        #                     count += 1
        #     return count


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
        
        idxs = np.random.choice(len(self.buffer), size= batch_size, replace=False)
        
        #selected_idxs = np.concatenate([high_reward_idxs, idxs])
        batch = [self.buffer[i] for i in idxs]
        states, actions_idx, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions_idx), list(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
