# simulare partite contro altri bot 
# allenare il tuo agente
# salvare i pesi del modello (model.save())

import numpy as np
from importlib import import_module
from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import RandomPlayer, AgressivePlayer, WeakerPlayerAttacker
from rl_agent import RLDicewarsAgent
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from collections import deque


NUM_EPISODES = 1000
SAVE_MODEL_PATH = "./saved_models/dicewars_rl_model.h5"

# Crea directory se non esiste
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

def calculate_step_reward(prev_state, new_state, player_idx):
    """
    Reward intermedio basato su: conquiste, perdite, crescita di dadi
    """
    reward = 0

    # Aree controllate
    prev_areas = len(prev_state.player_areas[player_idx])
    new_areas = len(new_state.player_areas[player_idx])
    reward += (new_areas - prev_areas) * 0.3

    # Dadi totali
    prev_dice = prev_state.player_num_dice[player_idx]
    new_dice = new_state.player_num_dice[player_idx]
    reward += (new_dice - prev_dice) * 0.05

    # Penalità se ha fatto "end turn" senza attaccare
    if prev_areas == new_areas and prev_dice == new_dice:
        reward -= 0.2

    return reward

def calculate_final_reward(winner, player_idx):
    return 1.0 if winner == player_idx else -0.5


# Training loop
agent = RLDicewarsAgent()

## plots
win_history = []
reward_history = []
moving_avg = deque(maxlen=50)  # media mobile su ultimi 50 episodi

plt.ion()  # interactive mode
fig, ax = plt.subplots(figsize=(8, 4))
line1, = ax.plot([], [], label='Win Rate', color='blue')
line2, = ax.plot([], [], label='Avg Reward', color='orange')
ax.set_xlim(0, NUM_EPISODES)
ax.set_ylim(-1, 1.1)
ax.set_title("Training Progress")
ax.set_xlabel("Episode")
ax.set_ylabel("Value")
ax.legend()


## training loop
for episode in range(NUM_EPISODES):
    players = [agent] + [RandomPlayer() for _ in range(3)]
    game = Game(num_seats=4)
    match = Match(game)
    grid, state = match.game.grid, match.state

    history = []

    while match.winner == -1:
        current_player = state.player
        prev_state = state  # <- snapshot prima dell'azione

        if current_player == 0:
            action = agent.select_action(grid, state)
            state_vec = agent.encode_state(grid, state)
        else:
            action = players[current_player].get_attack_areas(grid, state)

        grid, state = match.step(action)

        if current_player == 0:
            reward = calculate_step_reward(prev_state, state, player_idx=0)
            history.append((state_vec, action, reward))


    # Final reward da partita
    final_reward = calculate_final_reward(match.winner, player_idx=0)
    episode_reward = sum([r for _, _, r in history]) + final_reward
    reward_history.append(episode_reward)

    # Registra vincita
    won = 1 if match.winner == 0 else 0
    win_history.append(won)
    moving_avg.append(won)

    # Allena
    for state_vec, action_taken, step_reward in history:
        total_reward = step_reward + final_reward
        agent.train_step(state_vec, action_taken, total_reward)

    # Salva modello ogni 50 episodi
    if (episode + 1) % 50 == 0:
        agent.save_model()

    # === AGGIORNA IL GRAFICO ===
    line1.set_data(range(len(win_history)), [np.mean(win_history[max(0, i-50):i+1]) for i in range(len(win_history))])
    line2.set_data(range(len(reward_history)), reward_history)
    ax.set_xlim(0, len(win_history) + 10)
    ax.set_ylim(-1, max(1.1, max(reward_history, default=1)))
    plt.pause(0.01)
    
plt.ioff()
plt.show()
