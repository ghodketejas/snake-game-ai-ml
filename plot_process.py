import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import re

def plot_listener(queue, plot_path, visual_debug, model_name):
    """
    Listen for score updates from the training process and update/save the plot in real time.
    """
    if not visual_debug:
        import matplotlib
        matplotlib.use('Agg')
    scores = []
    mean_scores = []
    fig, ax = plt.subplots()
    # Print backend for debugging
    print(f"Matplotlib backend: {plt.get_backend()}")
    if visual_debug:
        plt.ion()
        # In debug mode, set window always on top for Tkinter or Qt
        try:
            fig_manager = plt.get_current_fig_manager()
            # For Qt backend
            if hasattr(fig_manager.window, 'setWindowFlag'):
                from PyQt5 import QtCore
                fig_manager.window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                fig_manager.window.show()
            # For Tk backend
            elif hasattr(fig_manager.window, 'wm_attributes'):
                fig_manager.window.wm_attributes('-topmost', 1)
        except Exception as e:
            print(f"Window stacking control not supported: {e}")
    # Use model_name for chart title
    run_label = model_name
    while True:
        data = queue.get()
        if data == 'DONE':
            # Set title to just the model name and redraw
            ax.set_title(run_label)
            if visual_debug:
                plt.draw()
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
        if visual_debug:
            plt.draw()
            plt.pause(0.001)
    plt.close(fig)

if __name__ == '__main__':
    # Example usage for manual testing
    queue = mp.Queue()
    import sys
    plot_path = sys.argv[1] if len(sys.argv) > 1 else 'plot.png'
    visual_debug = sys.argv[2] == 'True' if len(sys.argv) > 2 else True
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'model'
    plot_listener(queue, plot_path, visual_debug, model_name) 