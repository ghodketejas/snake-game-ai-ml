"""
Classic DQN agent for Snake Game AI training.

This module implements a Deep Q-Network (DQN) agent based on the original
Patrick Loeber implementation, with improvements for better training stability
and reproducibility.
"""

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import (
    plot, ask_visual_debug, get_next_save_paths, get_device, 
    print_device_info, write_training_csv, ask_visual_debug_auto, ask_show_pygame
)
import multiprocessing as mp

# Training hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """
    Classic DQN agent for Snake game training.
    
    Implements a Deep Q-Network with experience replay and epsilon-greedy
    exploration strategy. Uses a simple state representation and basic
    reward structure for learning Snake gameplay.
    """
    
    def __init__(self, device=None):
        """
        Initialize the agent with neural network and experience replay memory.
        
        Args:
            device: PyTorch device for computation (auto-detected if None)
        """
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.device = device if device is not None else get_device()
        self.model = Linear_QNet(11, 256, 3, device=self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

    def get_state(self, game):
        """
        Extract current game state as a feature vector for the agent.
        
        Features include:
        - Danger detection in three directions (straight, right, left)
        - Current movement direction (one-hot encoded)
        - Food location relative to snake head
        
        Args:
            game: SnakeGameAI instance representing current game state
            
        Returns:
            numpy.ndarray: 11-dimensional state vector
        """
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            # Move direction (one-hot)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location relative to head
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in experience replay memory.
        
        Args:
            state: Current state vector
            action: Action taken (one-hot encoded)
            reward: Reward received
            next_state: Next state vector
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train on a batch of experiences from memory (experience replay).
        
        Samples BATCH_SIZE experiences from memory and performs
        a training step on the batch.
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
        
        Args:
            state: Current state vector
            action: Action taken (one-hot encoded)
            reward: Reward received
            next_state: Next state vector
            done: Whether episode ended
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Epsilon decreases linearly with the number of games played,
        transitioning from exploration to exploitation.
        
        Args:
            state: Current state vector
            
        Returns:
            list: One-hot encoded action [straight, right, left]
        """
        self.epsilon = 80 - self.n_games  # Linear epsilon decay
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            # Explore: random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: choose best action from model
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move


def train(visual_debug=None, num_games=None, speed=None, custom_name=None):
    """
    Main training loop for the classic DQN agent.
    
    Handles user input, training execution, and result saving. Supports
    both interactive and automated training modes.
    
    Args:
        visual_debug: Whether to show training visualization
        num_games: Number of games to train for
        speed: Game speed (FPS)
        custom_name: Custom prefix for output files
    """
    import os
    
    # Configure Pygame display based on user preference
    show_pygame = ask_show_pygame()
    if not show_pygame:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        speed = 500  # Maximum speed for headless training

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    device = get_device()
    agent = Agent(device=device)
    model_name, plot_path, model_save_path, csv_path = get_next_save_paths('model', custom_name=custom_name)
    
    # Get training parameters
    if visual_debug is False:
        visual_debug, num_games, speed = ask_visual_debug_auto(visual_debug, num_games, speed)
    elif visual_debug is not None and num_games is not None and speed is not None:
        visual_debug, num_games, speed = ask_visual_debug_auto(visual_debug, num_games, speed)
    else:
        visual_debug, num_games, speed = ask_visual_debug()

    # Configure game speed
    import game as _game_module
    _game_module.SPEED = speed

    DELAY_PER_MOVE = 0
    game = SnakeGameAI(delay_per_move=DELAY_PER_MOVE)
    print_device_info(device)
    
    n_games = 0
    csv_rows = []
    queue = None
    plotter = None
    
    try:
        # Initialize plotting if in debug mode
        if visual_debug:
            import plot_process
            queue = mp.Queue()
            plotter = mp.Process(target=plot_process.plot_listener, args=(queue, plot_path, visual_debug, model_name))
            plotter.start()
            queue.put(([], []))
        
        # Main training loop
        while n_games < num_games:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                game.reset()
                agent.n_games += 1
                n_games += 1
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    agent.model.save(model_save_path)
                    
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                if visual_debug and queue is not None:
                    queue.put((plot_scores.copy(), plot_mean_scores.copy()))
                csv_rows.append([agent.n_games, score, record])
                
    finally:
        # Cleanup and save results
        if visual_debug and queue is not None:
            queue.put('DONE')
            
        agent.model.save(model_save_path)
        
        if visual_debug and plotter is not None:
            plotter.join()
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from helper import plot
            plot(plot_scores, plot_mean_scores, model_name=model_name, show_window=False)
            plt.savefig(plot_path)
            plt.close()
            
        write_training_csv(csv_path, model_name, csv_rows)
        print(f"Final model saved to {model_save_path}")
        import pygame
        pygame.display.quit()


if __name__ == '__main__':
    train()