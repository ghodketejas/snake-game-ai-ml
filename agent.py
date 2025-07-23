import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import multiprocessing as mp
import subprocess
import os
import re
import tkinter as tk
from tkinter import messagebox
import sys

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def get_next_model_name(debug=False):
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    existing = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    if debug:
        nums = [int(re.findall(r'debug_model(\d+)\.pth', f)[0]) for f in existing if re.match(r'debug_model\d+\.pth', f)]
        next_num = max(nums) + 1 if nums else 1
        model_name = f"debug_model{next_num}"
        plot_path = os.path.join('results', f"{model_name}_plot.png")
        model_save_path = os.path.join(model_dir, f"{model_name}.pth")
    else:
        nums = [int(re.findall(r'model(\d+)\.pth', f)[0]) for f in existing if re.match(r'model\d+\.pth', f)]
        next_num = max(nums) + 1 if nums else 1
        model_name = f"model{next_num}"
        plot_path = os.path.join('results', f"{model_name}_plot.png")
        model_save_path = os.path.join(model_dir, f"{model_name}.pth")
    return model_name, plot_path, model_save_path

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.device = get_device()
        self.model = Linear_QNet(11, 256, 3, device=self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
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
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return final_move


def ask_visual_debug():
    result = {'choice': None}
    def set_debug():
        result['choice'] = True
        root.destroy()
    def set_train():
        result['choice'] = False
        root.destroy()
    def on_close():
        root.destroy()
        sys.exit(0)
    root = tk.Tk()
    root.title('Snake Game AI - Select Mode')
    root.geometry('500x250')
    root.configure(bg='#f0f0f0')
    root.eval('tk::PlaceWindow . center')
    root.protocol('WM_DELETE_WINDOW', on_close)
    label = tk.Label(root, text='How do you want to run the training?', font=('Arial', 14, 'bold'), bg='#f0f0f0')
    label.pack(pady=(30, 18))
    desc = tk.Label(root, text='Choose "Debug" to see the live chart and slow down the game,\nor "Train" for fastest training with no chart window.', font=('Arial', 11), bg='#f0f0f0')
    desc.pack(pady=(0, 32))
    btn_frame = tk.Frame(root, bg='#f0f0f0')
    btn_frame.pack()
    debug_btn = tk.Button(btn_frame, text='Debug (Show Chart)', width=18, height=2, font=('Arial', 11, 'bold'), bg='#4caf50', fg='white', command=set_debug)
    debug_btn.pack(side='left', padx=20)
    train_btn = tk.Button(btn_frame, text='Train (No Chart)', width=18, height=2, font=('Arial', 11, 'bold'), bg='#2196f3', fg='white', command=set_train)
    train_btn.pack(side='right', padx=20)
    root.mainloop()
    return result['choice']

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # Use popup to ask user for visual debug
    visual_debug = ask_visual_debug()
    DELAY_PER_MOVE = 0.05 if visual_debug else 0
    game = SnakeGameAI(delay_per_move=DELAY_PER_MOVE)

    # Print device info
    if agent.device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(agent.device)
        print(f"Using device: {agent.device} ({gpu_name})")
    else:
        import platform
        cpu_name = platform.processor() or platform.uname().processor or 'CPU'
        print(f"Using device: {agent.device} ({cpu_name})")

    # Get unique model name, plot path, and model save path
    model_name, plot_path, model_save_path = get_next_model_name(debug=visual_debug)

    # Start plotting process, pass plot_path and visual_debug and model_name
    queue = mp.Queue()
    import plot_process
    plotter = mp.Process(target=plot_process.plot_listener, args=(queue, plot_path, visual_debug, model_name))
    plotter.start()
    # Send initial empty update to show chart window immediately
    if visual_debug:
        queue.put(([], []))

    MAX_GAMES = 1000
    try:
        while agent.n_games < MAX_GAMES:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(model_save_path)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                # Send data to plotting process BEFORE printing to terminal
                queue.put((plot_scores.copy(), plot_mean_scores.copy()))
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
    finally:
        queue.put('DONE')
        # Always save the final model at the end
        agent.model.save(model_save_path)
        print(f"Final model saved to {model_save_path}")
        plotter.join()

if __name__ == '__main__':
    train()