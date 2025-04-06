import numpy as np
import tensorflow as tf
import os
import time
import random
import datetime
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import DefaultPlayer, AgressivePlayer, RandomPlayer, WeakerPlayerAttacker, PassivePlayer
from dicewars.grid import Grid
from rl_agent_2 import RLDicewarsAgent, ReplayBuffer


# Configurazione
NUM_EPISODES = 200
SAVE_MODEL_PATH = os.path.join("saved_models", "dicewars_rl_model_vsrandom_2.keras")
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

# Parametri di training
BUFFER_SIZE = 1000
BATCH_SIZE = 128
TRAIN_EVERY = 5  # Allena il modello ogni n episodi
EVAL_EVERY = 50  # Valuta l'agente ogni n episodi
SAVE_EVERY = 30  # Salva il modello ogni n episodi

# Creazione del buffer di replay e dell'agente RL
buffer = ReplayBuffer(max_size=BUFFER_SIZE)
agent = RLDicewarsAgent()

# Avversari (possono essere cambiati)
other_players = [RandomPlayer(), RandomPlayer(), RandomPlayer()]

# Variabili per il monitoraggio delle prestazioni
win_history = []
reward_history = []
moving_avg = deque(maxlen=100)  # Media mobile delle ultime 50 partite

def calculate_step_reward(prev_state, new_state, player_idx):
    """
    Reward intermedio basato su: conquiste, perdite, crescita di dadi.
    """
    reward = 0
    prev_areas = prev_state.player_num_areas[player_idx]
    new_areas = new_state.player_num_areas[player_idx]
    reward += (new_areas - prev_areas) * 0.04

    prev_dice = prev_state.player_num_dice[player_idx]/prev_areas
    new_dice = new_state.player_num_dice[player_idx]/new_areas
    reward += (new_dice - prev_dice) * 0.04
    
    prev_cl = prev_state.player_max_size[player_idx]
    new_cl = new_state.player_max_size[player_idx]
    reward += (new_cl - prev_cl) * 0.04
    
    # prev_alive = sum(1 for a in prev_state.player_num_areas if a > 0)
    # new_alive = sum(1 for a in new_state.player_num_areas if a > 0)
    
    # if new_alive < prev_alive:
    #     reward += 0.5
    
    
    # if prev_areas == new_areas and prev_cl == new_cl:
    #    reward -= 0.02

    return reward

def calculate_final_reward(winner, player_idx):
    return 10.0 if winner == player_idx else -5.0

def evaluate_agent(agent, num_matches=10, other_players=other_players):
    """
    Valuta l'agente in partite contro avversari fissi senza esplorazione (epsilon=0).
    """
    wins = 0
    for _ in range(num_matches):
        players = [agent] + other_players
        game = Game(num_seats=4)
        match = Match(game)
        grid, state = match.game.grid, match.state

        while match.winner == -1:
            current_player = state.player

            if current_player == 0:
                action_idx , action = agent.select_action(grid, state, epsilon=0.0)
            else:
                action = players[current_player].get_attack_areas(grid, state)

            grid, state = match.step(action)

        if match.winner == 0:
            wins += 1

    win_rate = wins / num_matches
    print(f"🎯 Evaluation win rate: {win_rate:.2f}")
    return win_rate

# Training loop
start_time = time.time()

for episode in tqdm(range(NUM_EPISODES), desc="Episode"):
    players = [agent] + other_players
    game = Game(num_seats=4)
    match = Match(game)
    grid, state = match.game.grid, match.state

    epsilon = max(0.01, 0.7 - episode / NUM_EPISODES)  # Decrescente

    history = []

    while match.winner == -1:
        current_player = state.player
        prev_state = state  

        if current_player == 0:
            action_idx, action = agent.select_action(grid, state, epsilon=epsilon)
            state_vec = agent.encode_state(grid, state)
        else:
            action = players[current_player].get_attack_areas(grid, state)

        grid, state = match.step(action)

        if current_player == 0:
            reward = calculate_step_reward(prev_state, state, player_idx=0)
            done = match.winner != -1
            state_vec = agent.encode_state(grid, prev_state)
            next_state_vec = agent.encode_state(grid, state)
            history.append((state_vec, action, reward))
            if done:
                buffer.add(state_vec, action_idx, action, reward + calculate_final_reward(match.winner, player_idx=0), next_state_vec, done)
            else:
                buffer.add(state_vec, action_idx, action, reward, next_state_vec, done)

    # Calcola il reward finale e aggiorna le statistiche
    final_reward = calculate_final_reward(match.winner, player_idx=0)
    episode_reward = sum(r for _, _, r in history) + final_reward
    reward_history.append(episode_reward)

    won = 1 if match.winner == 0 else 0
    win_history.append(won)
    moving_avg.append(won)

    # Allena il modello se il buffer ha abbastanza campioni
    if (episode + 1) % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
        states, actions_idx, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        agent.train_batch(states, actions_idx, actions, rewards, next_states, dones)

    # Stampa informazioni di progresso
    print(f"Episode {episode + 1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Win Rate: {np.mean(moving_avg):.2f} | Winner: {'Agent' if match.winner == 0 else 'Opponent'}")

    # Salva il modello periodicamente
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save_model()

    # Valutazione periodica dell'agente
    if (episode + 1) % EVAL_EVERY == 0:
        eval_win_rate = evaluate_agent(agent, num_matches=10, other_players=other_players)

# Stampa il tempo totale di esecuzione
print("Training completed in", round(time.time() - start_time, 2), "seconds")
