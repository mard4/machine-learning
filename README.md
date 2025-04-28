# Dice Wars AI player

The goal of this project is to build an artificial intelligence (AI) that is able to play the 'Dice Wars' game. To be able to do that we use Reinforcement Learning.
<p align="center">
  <a href="https://github.com/user-attachments/files/19937816/report_ml.pdf">
    <img src="https://github.com/user-attachments/assets/8a51d1fe-2533-43ab-a182-ae7663aabfd3" width="400px"><br>
    <b>Report</b>
  </a>
</p>


# Dice Wars

Dice Wars is a turn-based strategy game where players aim to conquer all territories using dice.  
Each player takes turns attacking neighboring enemy areas from their own territories that have more than one die.  
Both attacker and defender roll all their dice — the player with the higher total wins.

---

# AI Agent

Our agent for Dice Wars follows a Deep Q-Learning (DQN) approach, illustrated in the figure below.

At the heart of the agent is a neural network that learns to approximate the Q-value function, **Q(s, a)**, which estimates the long-term expected reward of taking an action **a** in a given state **s**, assuming the agent acts optimally from that point forward.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ab18ef37-d033-4614-bbdb-b57c45fa77e7" width="500px">
</p>

The choice of a DQN was motivated by the structure of Dice Wars: a turn-based game with a discrete and sequential environment.  
The game involves a finite number of moves per turn, with discrete actions represented by "attack" or "end turn" decisions, and clearly defined game states (the board grid and match status).

This structure makes Q-learning a natural fit, as it thrives in discrete environments where each step depends heavily on previous choices.

---

## Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a9092b3-0b8c-45da-a556-36362838f65c" width="450px">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9924a7f-ad23-44af-9a32-321aa475cdb0" width="450px">
</p>
