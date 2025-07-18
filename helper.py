import matplotlib.pyplot as plt
# Removed: from IPython import display

plt.ion()

def plot(scores, mean_scores):
    # Removed: display.clear_output(wait=True)
    # Removed: display.display(plt.gcf())
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
    plt.pause(0.001)
