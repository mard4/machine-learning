##log_dir = "logs/dicewars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import numpy as np
from tensorflow.summary import create_file_writer
import datetime
from importlib import import_module
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import RandomPlayer, AgressivePlayer, WeakerPlayerAttacker, PassivePlayer
from dicewars.grid import Grid
from rl_agent import RLDicewarsAgent
from rl_agent import ReplayBuffer
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
#import wandb
import time
from datetime import datetime

#tf.config.list_physical_devices('GPU')

start_time = time.time()

# run = wandb.init(
#     project='Machine-Learning-DICEWARS',
#     name= f"DiceWars_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  
# )
# run_id = run.id 
# api = wandb.Api()
# run = api.run(f"Machine-Learning-DICEWARS/{run_id}")

if tf.config.list_physical_devices('GPU'):
    print("using cuda ok")
else:
    print('using cpu')

NUM_EPISODES = 1000
SAVE_MODEL_PATH = "D:/PhD utwente/courses/Machine learning/Exercises/dicewars-env-v1/saved_models/dicewars_rl_model.keras"
#SAVE_MODEL_PATH = "./saved_models/dicewars_rl_model_new.keras"
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

## Buffer
buffer = ReplayBuffer(max_size=10000)
BATCH_SIZE = 128
TRAIN_EVERY = 10  # ogni 10 episodi

#other_players = [PassivePlayer(), WeakerPlayerAttacker(), WeakerPlayerAttacker()]
other_players = [RandomPlayer(), RandomPlayer(), RandomPlayer()]


def calculate_step_reward(prev_state, new_state, player_idx):
    """
    Reward intermedio basato su: conquiste, perdite, crescita di dadi
    """
    reward = 0

    # Aree controllate
    prev_areas = len(prev_state.player_areas[player_idx])
    new_areas = len(new_state.player_areas[player_idx])
    reward += (new_areas - prev_areas) * 0.4

    # Dadi totali
    prev_dice = prev_state.player_num_dice[player_idx]
    new_dice = new_state.player_num_dice[player_idx]
    reward += (new_dice - prev_dice) * 0.04

    # Penalità se ha fatto "end turn" senza attaccare
    if prev_areas == new_areas and prev_dice == new_dice:
       reward -= 0.2

    return reward

def calculate_final_reward(winner, player_idx):
    return 20.0 if winner == player_idx else -10.0

def evaluate_agent(agent, num_matches=10, other_players=other_players):
    wins = 0

    for _ in range(num_matches):
        players = [agent] + other_players
        game = Game(num_seats=4)
        match = Match(game)
        grid, state = match.game.grid, match.state

        while match.winner == -1:
            current_player = state.player

            if current_player == 0:
                action = agent.select_action(grid, state, epsilon=0.0)  # no esplorazione
            else:
                action = players[current_player].get_attack_areas(grid, state)

            grid, state = match.step(action)

        if match.winner == 0:
            wins += 1

    win_rate = wins / num_matches
    print(f"🎯 Evaluation match win rate (no epsilon): {win_rate:.2f}")
    return win_rate


# Training loop
agent = RLDicewarsAgent()

## plots
win_history = []
reward_history = []
moving_avg = deque(maxlen=50)  # media mobile su ultimi 50 episodi


## training loop
for episode in tqdm(range(NUM_EPISODES), desc="Episode"):
    players = [agent] + other_players
    game = Game(num_seats=4)
    match = Match(game)
    grid, state = match.game.grid, match.state
    ##print("→ Player order:", [type(p).__name__ for p in players])
    epsilon =max(0.01, 0.1 - episode / NUM_EPISODES)  # decrescente

    history = []

    while match.winner == -1:
        current_player = state.player
        prev_state = state  # <- snapshot prima dell'azione
        
        if current_player == 0:
            action = agent.select_action(grid, state, epsilon=epsilon) # rimiuovi epsilon se non vuoi usare epsilon greedy
            state_vec = agent.encode_state(grid, state)
        else:
            action = players[current_player].get_attack_areas(grid, state)

        grid, state = match.step(action)

        if current_player == 0:
            #q_values = agent.model.predict(state_vec[None, :])[0]
            #print(f"Q-values per lo stato corrente: {q_values}")
            reward = calculate_step_reward(prev_state, state, player_idx=0)
            done = match.winner != -1
            state_vec = agent.encode_state(grid, prev_state)
            next_state_vec = agent.encode_state(grid,state)
            history.append((state_vec, action, reward))
            buffer.add(state_vec, action, reward, next_state_vec, done)


    # Final reward da partita
    final_reward = calculate_final_reward(match.winner, player_idx=0)
    episode_reward = sum([r for _, _, r in history]) + final_reward
    reward_history.append(episode_reward)

    # Registra vincita
    won = 1 if match.winner == 0 else 0
    win_history.append(won)
    moving_avg.append(won)


    # Allena senza buffer
    #for state_vec, action_taken, step_reward in history:
    #    total_reward = step_reward + final_reward
    #    agent.train_step(state_vec, action_taken, total_reward)
    
    ## Buffer training
    if (episode + 1) % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        agent.train_batch(states, actions, rewards, next_states, dones)


    print(f"Episode {episode + 1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Win Rate: {np.mean(moving_avg):.2f} | 🏆 Winner: {players[match.winner].__class__.__name__}")
    #wandb.log({'Episode': episode , 'Reward': episode_reward, 'Win Rate': np.mean(moving_avg), 'Epsilon': (epsilon)})
    
    # Salva modello ogni 20 episodi
    if (episode + 1) % 30 == 0:
        agent.save_model()
    if (episode + 1) % 50 == 0:
        eval_win_rate = evaluate_agent(agent, other_players=other_players)
        #wandb.log({'eval_win_rate': eval_win_rate, 'episode': episode})
      
        
print("Time", time.time() - start_time, "seconds")