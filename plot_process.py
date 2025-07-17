import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import re

plt.ion()

def plot_listener(queue, plot_path):
    scores = []
    mean_scores = []
    fig, ax = plt.subplots()
    # Extract run/model number from plot_path
    match = re.search(r'(model\d+)_plot', os.path.basename(plot_path))
    run_label = match.group(1) if match else 'Run'
    while True:
        data = queue.get()
        if data == 'DONE':
            # Save final plot
            fig.savefig(plot_path)
            break
        scores, mean_scores = data
        ax.clear()
        ax.set_title(f'Training Progress for {run_label}')
        ax.set_xlabel('Number of Games')
        ax.set_ylabel('Score')
        ax.plot(scores)
        ax.plot(mean_scores)
        ax.set_ylim(bottom=0)
        if scores:
            ax.text(len(scores)-1, scores[-1], str(scores[-1]))
        if mean_scores:
            ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.draw()
        plt.pause(0.001)
    plt.close(fig)

if __name__ == '__main__':
    queue = mp.Queue()
    import sys
    plot_path = sys.argv[1] if len(sys.argv) > 1 else 'plot.png'
    plot_listener(queue, plot_path) 