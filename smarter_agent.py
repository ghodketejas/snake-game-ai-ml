"""
smarter_agent.py
A smarter Snake agent designed to outperform the baseline agent (agent.py) with improved learning speed and efficiency.
"""

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from smarter_model import DeeperQNet, SmarterQTrainer
from helper import plot, ask_visual_debug, get_next_save_paths, get_device, print_device_info, write_training_csv
import multiprocessing as mp
import os
import re

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class SmarterAgent:
    """
    Smarter DQN agent for Snake. Uses enhanced state, reward shaping, and a deeper neural network.
    """
    def __init__(self, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, lr=LR * 0.5):
        """
        Initialize the smarter agent, deeper neural network, and experience replay memory.
        """
        self.n_games = 0
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.gamma = 0.95  # Higher discount for longer-term rewards
        self.memory = deque(maxlen=MAX_MEMORY)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeeperQNet(21, [512, 256, 128], 3, device=self.device, dropout_rate=0.2)
        self.trainer = SmarterQTrainer(
            self.model, lr=lr, gamma=self.gamma, device=self.device, clip_grad_norm=1.0
        )

    def get_state(self, game):
        """
        Returns an enhanced state vector for the smarter agent.
        Features include distances to walls/food, food alignment, snake length, and strategic features.
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        dist_to_left_wall = head.x
        dist_to_right_wall = game.w - head.x
        dist_to_top_wall = head.y
        dist_to_bottom_wall = game.h - head.y
        
        dist_to_food_x = abs(game.food.x - head.x)
        dist_to_food_y = abs(game.food.y - head.y)
        dist_to_food = dist_to_food_x + dist_to_food_y
        
        food_same_row = game.food.y == head.y
        food_same_col = game.food.x == head.x
        
        snake_length = len(game.snake)
        
        # Danger detection
        danger_straight = (dir_r and game.is_collision(point_r)) or \
                         (dir_l and game.is_collision(point_l)) or \
                         (dir_u and game.is_collision(point_u)) or \
                         (dir_d and game.is_collision(point_d))
        danger_right = (dir_u and game.is_collision(point_r)) or \
                      (dir_d and game.is_collision(point_l)) or \
                      (dir_l and game.is_collision(point_u)) or \
                      (dir_r and game.is_collision(point_d))
        danger_left = (dir_d and game.is_collision(point_r)) or \
                     (dir_u and game.is_collision(point_l)) or \
                     (dir_r and game.is_collision(point_u)) or \
                     (dir_l and game.is_collision(point_d))
        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
            dist_to_food_x / game.w,
            dist_to_food_y / game.h,
            dist_to_left_wall / game.w,
            dist_to_right_wall / game.w,
            dist_to_top_wall / game.h,
            food_same_row,
            food_same_col,
            snake_length / 100.0,
            dist_to_food < 40,
            len(game.snake) > 10
        ]
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in memory for experience replay.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train on a batch of experiences from memory (experience replay).
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train on the most recent experience (single step update).
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy with exponential decay.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** self.n_games))
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            # Explore: random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: choose best action from model
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1
        return final_move

    def calculate_enhanced_reward(self, game, reward, done, old_distance_to_food=None):
        """
        Enhanced reward shaping to encourage better learning:
        - Reward for getting closer to food
        - Penalty for moving away
        - Small penalty per step
        - Bonus for being close to food
        - Penalty for being near wall with long snake
        - Extra penalty for dying with long snake
        """
        enhanced_reward = reward
        if not done:
            current_distance = abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y)
            if old_distance_to_food is not None:
                if current_distance < old_distance_to_food:
                    enhanced_reward += 1
                elif current_distance > old_distance_to_food:
                    enhanced_reward -= 0.5
            enhanced_reward -= 0.1  # Small penalty per step
            if current_distance <= 40:
                enhanced_reward += 0.5
            if len(game.snake) > 10:
                wall_distance = min(game.head.x, game.head.y, game.w - game.head.x, game.h - game.head.y)
                if wall_distance <= 40:
                    enhanced_reward -= 0.3
        else:
            if len(game.snake) > 10:
                enhanced_reward -= 5
        return enhanced_reward


def train():
    """
    Main training loop for the smarter agent. Handles user input, training, and saving results.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = SmarterAgent()
    model_name, plot_path, model_save_path, csv_path = get_next_save_paths('smarter_model')
    visual_debug, num_games, speed = ask_visual_debug()
    DELAY_PER_MOVE = 0.05 if visual_debug else 0
    game = SnakeGameAI(delay_per_move=DELAY_PER_MOVE)
    game.clock.tick(speed)
    print_device_info(agent.device)
    queue = mp.Queue()
    import plot_process
    plotter = mp.Process(target=plot_process.plot_listener, args=(queue, plot_path, visual_debug, model_name))
    plotter.start()
    if visual_debug:
        queue.put(([], []))
    MAX_GAMES = num_games
    csv_rows = []
    try:
        while agent.n_games < MAX_GAMES:
            state_old = agent.get_state(game)
            old_distance_to_food = abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            enhanced_reward = agent.calculate_enhanced_reward(game, reward, done, old_distance_to_food)
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, final_move, enhanced_reward, state_new, done)
            agent.remember(state_old, final_move, enhanced_reward, state_new, done)
            if done:
                game.reset()
                game.clock.tick(speed)
                agent.n_games += 1
                agent.train_long_memory()
                if score > record:
                    record = score
                    agent.model.save(model_save_path)
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                queue.put((plot_scores.copy(), plot_mean_scores.copy()))
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                csv_rows.append([agent.n_games, score, record])
    finally:
        # Always save model, plot, and CSV, even if interrupted
        queue.put('DONE')
        agent.model.save(model_save_path)
        write_training_csv(csv_path, model_name, csv_rows)
        print(f"Final model saved to {model_save_path}")
        plotter.join()

if __name__ == '__main__':
    train()