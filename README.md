# Snake Game AI (Enhanced Version)

>This repository is a clone of the [patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch) project by Patrick Loeber, as featured on the freeCodeCamp YouTube channel. All original code and ideas are credited to Patrick Loeber and freeCodeCamp.

## Objective

The main objective of this clone is to experiment with and improve the AI logic for the Snake game, and to provide a robust, reproducible training and evaluation workflow.

## Key Features & Improvements

- **Fresh Model Per Run:**  
  Every time you run `agent.py`, a new model is trained from scratch and saved as `model/modelX.pth` (where X is an incrementing number).
- **Result Plotting:**  
  After each training session, a plot of the training progress (scores and mean scores) is saved as `results/modelX_plot.png`, matching the model number.
- **Multiprocessing Plotting:**  
  The training progress plot updates in real time in a separate window, allowing you to interact with both the game and the plot without crashes.
- **Grid-Aligned Snake and Food:**  
  The snake's head and food are always aligned to the grid, ensuring correct collision and scoring behavior.
- **Bug Fixes & Robustness:**  
  - All positions are integers to avoid rendering issues.
  - Model saving and directory creation are robust and error-free.
  - Code is compatible with Windows multiprocessing requirements.
- **Human Play Mode:**  
  Play the game yourself using `snake_game_human.py`.

## Project Structure

- `game.py` - Core Snake game logic and environment (AI version)
- `agent.py` - AI agent and training loop (run this to train a new model)
- `model.py` - Neural network model definition and training logic
- `helper.py` - Utility functions for plotting (used in legacy mode)
- `plot_process.py` - Real-time plotting process for training progress
- `snake_game_human.py` - Play the game as a human
- `model/` - Folder containing all trained models (`model1.pth`, `model2.pth`, ...)
- `results/` - Folder containing all training plots (`model1_plot.png`, `model2_plot.png`, ...)
- `arial.ttf` - Font for rendering text

## How to Use

1. **Train a New Model**
   - Run `python agent.py`
   - A new model will be trained from scratch and saved as `model/modelX.pth`
   - The training progress plot will be saved as `results/modelX_plot.png`
   - The plot window updates in real time and can be moved/resized independently

2. **Play as a Human**
   - Run `python snake_game_human.py`

3. **Review Results**
   - All trained models are in the `model/` folder
   - All training plots are in the `results/` folder

## Credits

- Original project and tutorial by [Patrick Loeber](https://github.com/patrickloeber/snake-ai-pytorch)
- YouTube video: [AI Learns to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds&t=2355s)
- Hosted on [freeCodeCamp](https://www.freecodecamp.org/)

## Next Steps

You can further experiment with the AI logic, try new reward strategies, or tune hyperparameters to improve performance.  
Feel free to fork and contribute! 

## Benchmarking & Results

**Benchmark Settings:**
- Speed: 100
- Width: 750
- Height: 750
- Max Games: 100
- MAX_MEMORY: 100,000
- BATCH_SIZE: 1,000
- LR: 0.001

### Test Results

| Test | Model File         | Plot File                | Notes/Observations |
|------|--------------------|--------------------------|--------------------|
| 1    | model/model1.pth   | results/model1_plot.png  | Max score: 82, Mean score: 7.89. The agent started learning after ~75 games, with a sharp increase in scores and high variance. |
| 2    | model/model2.pth   | results/model2_plot.png  | Max score: 62, Mean score: 4.23. The agent showed improvement after ~80 games, but the overall mean score was lower than run 1. |

### Example Plots

#### Test 1
![Test 1 Plot](results/model1_plot.png)

#### Test 2
![Test 2 Plot](results/model2_plot.png)

### Analysis

- **Learning Trends:**  
  In both runs, the agent showed little progress for the first 70–80 games, then rapidly improved, achieving much higher scores in the final games. This is typical for reinforcement learning, where the agent needs time to explore before it starts exploiting learned strategies.
- **Best Scores:**  
  - Run 1: 82 (mean: 7.89)
  - Run 2: 62 (mean: 4.23)
- **Observations:**  
  - There is significant variance in the scores after the agent starts learning, indicating that while the agent can achieve high scores, it is not yet fully consistent.
  - The mean score remains much lower than the best score, showing that high scores are occasional rather than regular.
  - The plot title is missing and displays as "training" due to a minor code issue (see below).

### Note on Plot Titles

The saved plots currently show the title as "training" or are missing a proper title. This is due to the code in `plot_process.py` using `ax.set_title('Training...')` or similar. To customize the plot title, I will update the plotting code to set a more descriptive title, such as `"Training Progress for Run X"`. 