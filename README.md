# Snake Game AI (Enhanced Version)

>This repository is a clone of the [patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch) project by Patrick Loeber, as featured on the freeCodeCamp YouTube channel. All original code and ideas are credited to Patrick Loeber and freeCodeCamp.

## Objective

The main objective of this clone is to experiment with and improve the AI logic for the Snake game, and to provide a robust, reproducible training and evaluation workflow.

## Key Features

- **Classic and Smarter Agents:**
  - `agent.py`: Baseline agent based on Patrick Loeber's DQN logic.
  - `smarter_agent.py`: Enhanced agent with richer state, improved reward shaping, and a deeper neural network.
- **Flexible Training:**
  - Every training run starts with a visual popup where you select debug/train mode, number of games, and game speed (recommended: 100 for slower gameplay).
  - Both agents support GPU acceleration if available.
- **Unique Model/Plot/CSV Saving:**
  - Each run saves its model, training plot, and CSV log with unique, sequential names for easy benchmarking.
- **Multiprocessing Plotting:**
  - In debug mode, a separate process handles live charting for smooth, crash-free visualization.
- **Consistent Board Size:**
  - The game board is always 750x750 for all training runs.
- **Human Play Mode:**
  - Play the game yourself using `snake_game_human.py`.

## How to Use

1. **Train a Model**
   - Run `python agent.py` or `python smarter_agent.py`.
   - A visual popup will always appear, allowing you to select debug or train mode, set the number of games, and set the game speed.
   - Models, plots, and CSV logs are saved in `model/` and `results/` subfolders.

2. **Play as a Human**
   - Run `python snake_game_human.py`.

3. **Review Results**
   - All trained models are in `model/`.
   - All plots are in `results/plots/`.
   - All training logs (CSV) are in `results/csv/`.

## Project Structure

- `game.py` - Core Snake game logic and environment
- `agent.py` - Baseline DQN agent and training loop
- `smarter_agent.py` - Enhanced agent and training loop
- `model.py` - Classic neural network model and trainer
- `smarter_model.py` - Enhanced model and trainer for the smarter agent
- `helper.py` - Utility functions (UI, plotting, file naming, etc.)
- `plot_process.py` - Multiprocessing plotting process
- `snake_game_human.py` - Human play mode
- `model/` - Trained models
- `results/plots/` - Training plots
- `results/csv/` - Training logs
- `arial.ttf` - Font for rendering text

## Benchmarking & Results

### Test Results Template

#### Patrick's model (agent.py):
| Test games | Model File | Plot File | CSV File | Notes/Observation |
|------------|------------|-----------|----------|-------------------|
| 100        |            |           |          |                   |
| 1000       |            |           |          |                   |
| cap (x)    |            |           |          |                   |

#### Smarter model (smarter_agent.py):
| Test games | Model File | Plot File | CSV File | Notes/Observation |
|------------|------------|-----------|----------|-------------------|
| 100        |            |           |          |                   |
| 1000       |            |           |          |                   |
| cap (x)    |            |           |          |                   |

### Analysis

- Summarize learning patterns, peak/mean scores, and key differences after benchmarking runs.

---

## Next Goal: User Input Logging for Supervised Learning

I plan to add a feature that records and stores user gameplay inputs in a dedicated folder. This will allow me to build a clean, high-quality dataset for future supervised learning experiments, enabling the training of models that can learn directly from human gameplay.

## Credits

- Original project and tutorial by [Patrick Loeber](https://github.com/patrickloeber/snake-ai-pytorch)
- YouTube video: [AI Learns to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds&t=2355s)
- Hosted on [freeCodeCamp](https://www.freecodecamp.org/)
