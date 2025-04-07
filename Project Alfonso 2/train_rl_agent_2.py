import numpy as np
import tensorflow as tf
import os
import time
import random
import datetime
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import csv

from dicewars.match import Match
from dicewars.game import Game
from dicewars.player import DefaultPlayer, AgressivePlayer, RandomPlayer, WeakerPlayerAttacker, PassivePlayer
from dicewars.grid import Grid
from rl_agent_2 import RLDicewarsAgent, ReplayBuffer
# CREAZIONE CARTELLA LOG
BASE_LOG_DIR = os.path.dirname("./")
CSV_LOG_DIR = os.path.join(BASE_LOG_DIR, "logs")
os.makedirs(CSV_LOG_DIR, exist_ok=True)
# CSV: Episodi
episode_log_file = open(os.path.join(CSV_LOG_DIR, "episode_logs.csv"), mode="w", newline='')
episode_writer = csv.writer(episode_log_file)
episode_writer.writerow(["Episode", "Reward", "Win", "MovingAvg", "Winner"])
# CSV: Descrittori
descriptor_log_file = open(os.path.join(CSV_LOG_DIR, "descriptor_logs.csv"), mode="w", newline='')
descriptor_writer = csv.writer(descriptor_log_file)
descriptor_writer.writerow(["Episode", "BorderStrength", "DiceAdvantage"])
# CSV: Q-values
qvalue_log_file = open(os.path.join(CSV_LOG_DIR, "qvalue_logs.csv"), mode="w", newline='')
qvalue_writer = csv.writer(qvalue_log_file)
qvalue_writer.writerow(["Episode", "Q_Max", "Q_Mean"])
# CSV: Evaluation win rate
evaluation_log_file = open(os.path.join(CSV_LOG_DIR, "evaluation_logs.csv"), mode="w", newline='')
evaluation_writer = csv.writer(evaluation_log_file)
evaluation_writer.writerow(["Episode", "EvalWinRate"])



# Configurazione
NUM_EPISODES = 400
SAVE_MODEL_PATH = os.path.join("saved_models", "dicewars_rl_model_vsrandom_3.keras")
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

# Parametri di training
BUFFER_SIZE = 1000
BATCH_SIZE = 128
TRAIN_EVERY = 10  # Allena il modello ogni n episodi
EVAL_EVERY = 40  # Valuta l'agente ogni n episodi
SAVE_EVERY = 30  # Salva il modello ogni n episodi
MAX_STEPS = 3000  # Limite di step per ogni match


# Creazione del buffer di replay e dell'agente RL
buffer = ReplayBuffer(max_size=BUFFER_SIZE)
agent = RLDicewarsAgent()

# Avversari (possono essere cambiati)
other_players = [RandomPlayer(), AgressivePlayer(), RandomPlayer()]

# Variabili per il monitoraggio delle prestazioni
win_history = []
reward_history = []
moving_avg = deque(maxlen=100)  # Media mobile delle ultime 50 partite

def calculate_step_reward(prev_state, new_state, player_idx,
                          took_action, valid_actions):
    """
    Reward intermedio basato su: conquiste, perdite, crescita di dadi.
    """
    reward = 0
    
    # calculate areas difference
    prev_areas = prev_state.player_num_areas[player_idx]
    new_areas = new_state.player_num_areas[player_idx]
    reward += (new_areas - prev_areas) * 0.04

    # calculate dice difference
    prev_dice = prev_state.player_num_dice[player_idx]/prev_areas
    new_dice = new_state.player_num_dice[player_idx]/new_areas
    reward += (new_dice - prev_dice) * 0.04
    
    # calculate cluster difference
    prev_cl = prev_state.player_max_size[player_idx]
    new_cl = new_state.player_max_size[player_idx]
    reward += (new_cl - prev_cl) * 0.04
    
    # prev_alive = sum(1 for a in prev_state.player_num_areas if a > 0)
    # new_alive = sum(1 for a in new_state.player_num_areas if a > 0)
    
    # if new_alive < prev_alive:
    #     reward += 0.5
    
    
    # if prev_areas == new_areas and prev_cl == new_cl:
    #    reward -= 0.02
    
    ## descriptors
    ## border
    border_strength = state_vec[-2]
    if border_strength < -2.0:
        reward -= 2.0

    # dice advantage
    advantage = state_vec[-1]  
    reward += advantage * 0.02
    
    ## penalty for losing areas
    if new_areas < prev_areas:
        lost = prev_areas - new_areas
        reward -= 0.1 * lost  
        
    ## cluster we alreadu have it but more reward??    
    if new_cl >= 10:
        reward += 0.2  # “Bravo! Hai un territorio grosso”
        
    if not took_action and len(valid_actions) > 1:
    # Aveva azioni ma ha passato
        reward -= 0.5


    return reward

def calculate_final_reward(winner, player_idx, forced_end=False):
    """
    If forced_end=True, la partita è stata interrotta perché troppi step.
    Possiamo dare una penalità extra a tutti in quell’evento.
    """
    if forced_end:
        # Partita "annullata": piccolo malus generico
        return -5.0
    else:
        return 10.0 if winner == player_idx else -5.0

def evaluate_agent(agent, num_matches=5, other_players=other_players):
    """
    Valuta l'agente in partite contro avversari fissi senza esplorazione (epsilon=0).
    """
    wins = 0
    for _ in range(num_matches):
        players = [agent] + other_players
        game = Game(num_seats=4)
        match = Match(game)
        grid, state = match.game.grid, match.state

        step_count = 0
        while match.winner == -1:
            step_count += 1
            # Limite step anche in valutazione (opzionale)
            if step_count >= MAX_STEPS:
                break
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
    
    step_count = 0
    forced_end = False  # Per capire se usciamo per eccesso di step

    while match.winner == -1:
        step_count += 1
        if step_count >= MAX_STEPS:
            # Partita non finisce entro un tot di turni
            forced_end = True
            break
        
        current_player = state.player
        prev_state = state  

        if current_player == 0:
            action_idx, action = agent.select_action(grid, state, epsilon=epsilon)
            state_vec = agent.encode_state(grid, state)
        else:
            action = players[current_player].get_attack_areas(grid, state)

        grid, state = match.step(action)

        if current_player == 0:
            valid_actions, _ = agent.get_valid_actions(grid, state)
            took_action = (action is not None)

    
        if current_player == 0:
            reward = calculate_step_reward(prev_state, state, player_idx=0,
                                           took_action=took_action, valid_actions=valid_actions)
            done = match.winner != -1
            state_vec = agent.encode_state(grid, prev_state)
            last_agent_state_vec = state_vec.copy()  # SALVA QUESTO PER I LOG SUCCESSIVI
            next_state_vec = agent.encode_state(grid, state)
            history.append((state_vec, action, reward))
            if done:
                buffer.add(state_vec, action_idx, action, reward + calculate_final_reward(match.winner, player_idx=0), next_state_vec, done)
            else:
                buffer.add(state_vec, action_idx, action, reward, next_state_vec, done)

    # Calcola il reward finale e aggiorna le statistiche
    final_reward = calculate_final_reward(match.winner, player_idx=0)
    step_rewards = [x[2] for x in history] if history else []
    episode_reward = sum(r for _, _, r in history) + final_reward
    reward_history.append(episode_reward)
    
    won = 1 if match.winner == 0 else 0
    win_history.append(won)
    moving_avg.append(won)
    
    ### === logs
    # Salva episodio
    episode_writer.writerow([
        episode + 1,
        round(episode_reward, 2),
        won,
        round(np.mean(moving_avg), 2),
        #"Agent" if match.winner == 0 else "Opponent",
        "Agent" if (match.winner == 0 and not forced_end) else "OpponentOrForced"

    ])
    
    
    ## === logs
    # Se abbiamo a disposizione l'ultimo state_vec dell'agente (non sempre salvato), puoi loggare border/adv
    # Per semplicità, recuperiamo la dimensione dal replayBuffer[-1], se esiste:
    if len(buffer.buffer) > 0:
        last_exp = buffer.buffer[-1]
        # last_exp: (state_vec, action_idx, action, reward, next_state_vec, done)
        if last_exp[0] is not None and len(last_exp[0]) >= 2:
            b_strength = round(last_exp[0][-2], 4)
            advantage = round(last_exp[0][-1], 4)
        else:
            b_strength = 0.0
            advantage = 0.0
    else:
        b_strength = 0.0
        advantage = 0.0

    descriptor_writer.writerow([
        episode + 1,
        b_strength,
        advantage
    ])
    
    
    
    # Salva descrittori (solo se sei current_player == 0)
    descriptor_writer.writerow([
        episode + 1,
        round(last_agent_state_vec[-2], 4),
        round(last_agent_state_vec[-1], 4)
    ])
    #### === logs

    # Allena il modello se il buffer ha abbastanza campioni
    if (episode + 1) % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
        states, actions_idx, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        loss = agent.train_batch(states, actions_idx, actions, rewards, next_states, dones)

        # Estrazione Q-values del primo stato
        q_values = agent.model.predict(states, verbose=0)[0]
        q_max = np.max(q_values)
        q_mean = np.mean(q_values)
        # Logga Q-values medi
        qvalue_writer.writerow([episode + 1, round(q_max, 4), round(q_mean, 4)])

    # Stampa informazioni di progresso
    print(f"Episode {episode + 1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Win Rate: {np.mean(moving_avg):.2f} | Winner: {'Agent' if match.winner == 0 else 'Opponent'}")

    # Salva il modello periodicamente
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save_model()

    # Valutazione periodica dell'agente
    if (episode + 1) % EVAL_EVERY == 0:
        print(f"📊 Running evaluation at episode {episode + 1}")
        try:
            eval_win_rate = evaluate_agent(agent, num_matches=10, other_players=other_players)
            evaluation_writer.writerow([episode + 1, round(eval_win_rate, 4)])
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

# Stampa il tempo totale di esecuzione
print("Training completed in", round(time.time() - start_time, 2), "seconds")
episode_log_file.close()
descriptor_log_file.close()
qvalue_log_file.close()
evaluation_log_file.close()
