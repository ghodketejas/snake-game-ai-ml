"""
Utility functions for Snake Game AI training and visualization.

This module provides helper functions for plotting, device management,
user interaction, file management, and training result logging.
"""

import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import re


def plot(scores, mean_scores, model_name=None, show_window=True):
    """
    Plot training scores and mean scores in real-time.
    
    Args:
        scores: List of individual game scores
        mean_scores: List of running mean scores
        model_name: Name of the model for plot title
        show_window: Whether to display the plot window
    """
    plt.clf()
    
    if model_name:
        if show_window:
            plt.title(f'{model_name} training...')
        else:
            plt.title(f'{model_name}')
    else:
        plt.title('Training...')
        
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        
    if show_window:
        plt.draw()
        try:
            plt.pause(0.001)
        except Exception:
            pass


def ask_visual_debug():
    """
    Show a popup dialog to select training mode and parameters.
    
    Returns:
        tuple: (visual_debug, num_games, speed)
    """
    result = {'choice': None, 'num_games': 1000, 'speed': 100}
    
    def set_debug():
        result['choice'] = True
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        speed = simpledialog.askinteger(
            'Game Speed',
            'Enter the game speed (recommended: 100 for slower gameplay):',
            initialvalue=100,
            minvalue=1,
            parent=root
        )
        if speed is None:
            root.destroy()
            sys.exit(0)
        result['speed'] = speed
        
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        num = simpledialog.askinteger(
            'Number of Games',
            'Enter the number of games to run:',
            initialvalue=1000,
            minvalue=1,
            parent=root
        )
        if num is None:
            root.destroy()
            sys.exit(0)
        result['num_games'] = num
        root.destroy()
        
    def set_train():
        result['choice'] = False
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        speed = simpledialog.askinteger(
            'Game Speed',
            'Enter the game speed (recommended: 100 for slower gameplay):',
            initialvalue=100,
            minvalue=1,
            parent=root
        )
        if speed is None:
            root.destroy()
            sys.exit(0)
        result['speed'] = speed
        
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        num = simpledialog.askinteger(
            'Number of Games',
            'Enter the number of games to run:',
            initialvalue=1000,
            minvalue=1,
            parent=root
        )
        if num is None:
            root.destroy()
            sys.exit(0)
        result['num_games'] = num
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
    return result['choice'], result['num_games'], result['speed']


def ask_visual_debug_auto(visual_debug, num_games, speed):
    """
    Automated version of ask_visual_debug for CI/benchmarking.
    
    Args:
        visual_debug: Whether to show visualization
        num_games: Number of games to train
        speed: Game speed
        
    Returns:
        tuple: (visual_debug, num_games, speed) unchanged
    """
    return visual_debug, num_games, speed


def ask_show_pygame():
    """
    Ask user whether to show Pygame window during training.
    
    Returns:
        bool: True if user wants to show window, False otherwise
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    show_game = messagebox.askyesno(
        'Show Game Window',
        'Do you want to display the Pygame window during training?\n(This may slow training down.)',
        parent=root
    )
    root.destroy()
    return bool(show_game)


def get_next_save_paths(prefix='model', custom_name=None):
    """
    Generate unique file paths for saving model, plot, and CSV results.
    
    Args:
        prefix: Base prefix for file names
        custom_name: Custom name to use instead of auto-generated
        
    Returns:
        tuple: (model_name, plot_path, model_save_path, csv_path)
    """
    model_dir = 'model'
    results_dir = 'results'
    plots_dir = os.path.join(results_dir, 'plots')
    csv_dir = os.path.join(results_dir, 'csv')
    
    # Create directories if they don't exist
    for d in [model_dir, results_dir, plots_dir, csv_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    if custom_name:
        model_name = custom_name
    else:
        # Find next available model number
        existing = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        pattern = re.compile(rf'{re.escape(prefix)}(\d+)\.pth')
        nums = [int(pattern.findall(f)[0]) for f in existing if pattern.match(f)]
        next_num = max(nums) + 1 if nums else 1
        model_name = f"{prefix}{next_num}"
        
    plot_path = os.path.join(plots_dir, f"{model_name}_plot.png")
    model_save_path = os.path.join(model_dir, f"{model_name}.pth")
    csv_path = os.path.join(csv_dir, f"{model_name}.csv")
    
    return model_name, plot_path, model_save_path, csv_path


def get_device():
    """
    Get the best available PyTorch device.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info(device):
    """
    Print detailed information about the device being used.
    
    Args:
        device: PyTorch device to print info for
    """
    import torch
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: {device} ({gpu_name})")
    else:
        import platform
        cpu_name = platform.processor() or platform.uname().processor or 'CPU'
        print(f"Using device: {device} ({cpu_name})")


def write_training_csv(csv_path, model_name, csv_rows):
    """
    Write training results to a CSV file.
    
    Args:
        csv_path: Path to save the CSV file
        model_name: Name of the model
        csv_rows: List of [game_number, score, record] rows
    """
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"Model: {model_name}"])
        writer.writerow(["Game", "Score", "Record"])
        writer.writerows(csv_rows)
    print(f"CSV log saved to {csv_path}")
