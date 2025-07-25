"""
benchmark_agents.py
Automated benchmarking and evaluation for Snake Game AI agents.
"""
import importlib
import os
import sys
import numpy as np

# Settings
AGENTS = [
    {"name": "Classic Agent", "module": "agent", "train_func": "train", "prefix": "model", "file_prefix": "bk_classic-agent"},
    {"name": "Smarter Agent", "module": "smarter_agent", "train_func": "train", "prefix": "smarter_model", "file_prefix": "bk_smart-agent"}
]
GAME_COUNTS = [100, 1000]
SPEED = 500
BOARD_SIZE = 750  # fixed in game.py
CAP_WINDOW = 100  # moving average window
CAP_PATIENCE = 500  # how many games to wait for improvement
CAP_THRESHOLD = 1.0  # min improvement to reset patience

RESULTS_DIRS = {
    "model": "model",
    "plot": os.path.join("results", "plots"),
    "csv": os.path.join("results", "csv")
}

def get_latest_file(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    return sorted(files)[-1] if files else "-"

def get_latest_csv_scores(csv_path):
    # Returns a list of scores from the latest csv file
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]  # skip header
            scores = [int(line.split(",")[1]) for line in lines if "," in line]
        return scores
    except Exception:
        return []

def run_train(agent_mod, train_func, visual_debug, num_games, speed, custom_name):
    # Call the agent's train function directly
    getattr(agent_mod, train_func)(visual_debug=visual_debug, num_games=num_games, speed=speed, custom_name=custom_name)
    # Do not call pygame.quit() here; let each agent handle its own cleanup if needed

def run_until_cap(agent_mod, train_func, prefix, agent_name):
    patience = 0
    best_avg = -float('inf')
    total_games = 0
    last_n_scores = []
    cap_scores = []
    cap_run = 0
    while True:
        cap_run += 1
        # Run in increments of 100 games, with unique name for each increment
        custom_name = f"benchmark_{prefix}_cap_{total_games+100}games"
        run_train(agent_mod, train_func, visual_debug=False, num_games=100, speed=SPEED, custom_name=custom_name)
        # Find the latest csv file for this agent
        csv_dir = RESULTS_DIRS["csv"]
        latest_csv = get_latest_file(csv_dir, custom_name)
        scores = get_latest_csv_scores(os.path.join(csv_dir, latest_csv))
        if not scores:
            break
        # Only consider new scores since last check
        new_scores = scores[len(cap_scores):]
        cap_scores.extend(new_scores)
        last_n_scores = cap_scores[-CAP_WINDOW:]
        if len(last_n_scores) < CAP_WINDOW:
            continue  # not enough data yet
        avg = np.mean(last_n_scores)
        if avg > best_avg + CAP_THRESHOLD:
            best_avg = avg
            patience = 0
        else:
            patience += 100
        if patience >= CAP_PATIENCE:
            break
        total_games += 100
    # After cap, return the total number of games and the last custom_name used
    return len(cap_scores), custom_name

def main():
    print("Snake AI Benchmarking Script (Automated)")
    print(f"Benchmarking settings: speed={SPEED}, board={BOARD_SIZE}x{BOARD_SIZE}, games={GAME_COUNTS} + cap mode\n")
    # Alternate between agents for each benchmark type
    for count in GAME_COUNTS:
        for agent in AGENTS:
            agent_mod = importlib.import_module(agent["module"])
            custom_name = f"{agent['file_prefix']}_{count}games"
            run_train(agent_mod, agent["train_func"], visual_debug=False, num_games=count, speed=SPEED, custom_name=custom_name)
    # Cap mode for each agent
    for agent in AGENTS:
        agent_mod = importlib.import_module(agent["module"])
        print(f"\n--- Benchmarking {agent['name']} until cap (early stopping) ---")
        total_games, cap_custom_name = run_until_cap(agent_mod, agent["train_func"], agent["prefix"], agent["name"])
        cap_file_prefix = f"{agent['file_prefix']}_cap_{total_games}games"
        model_file = get_latest_file(RESULTS_DIRS["model"], cap_file_prefix)
        plot_file = get_latest_file(RESULTS_DIRS["plot"], cap_file_prefix)
        csv_file = get_latest_file(RESULTS_DIRS["csv"], cap_file_prefix)
        # Remove the summary table printout at the end of main()

if __name__ == "__main__":
    main() 