import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import re

plt.ion()

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.draw()
    try:
        plt.pause(0.001)
    except Exception:
        pass

def ask_visual_debug():
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
        if speed is not None and speed > 0:
            result['speed'] = speed
        else:
            result['speed'] = 100
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
        if num is not None and num > 0:
            result['num_games'] = num
        else:
            result['num_games'] = 1000
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
        if speed is not None and speed > 0:
            result['speed'] = speed
        else:
            result['speed'] = 100
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
        if num is not None and num > 0:
            result['num_games'] = num
        else:
            result['num_games'] = 1000
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

def get_next_save_paths(prefix='model'):
    model_dir = 'model'
    results_dir = 'results'
    plots_dir = os.path.join(results_dir, 'plots')
    csv_dir = os.path.join(results_dir, 'csv')
    for d in [model_dir, results_dir, plots_dir, csv_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
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
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_device_info(device):
    import torch
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: {device} ({gpu_name})")
    else:
        import platform
        cpu_name = platform.processor() or platform.uname().processor or 'CPU'
        print(f"Using device: {device} ({cpu_name})")

def write_training_csv(csv_path, model_name, csv_rows):
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"Model: {model_name}"])
        writer.writerow(["Game", "Score", "Record"])
        writer.writerows(csv_rows)
    print(f"CSV log saved to {csv_path}")
