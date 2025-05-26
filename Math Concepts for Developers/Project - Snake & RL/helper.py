import matplotlib.pyplot as plt
from IPython import display

# Enable interactive mode
plt.ion() # To plot interactively

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf()) # Get current figure
    plt.clf() # Clear the current figure

    plt.title('Training...')
    plt.xlabel('Number of games')
    plt.ylabel('Score')

    plt.plot(scores, label='Scores', color='blue')
    plt.plot(mean_scores, label='Mean Scores', color='orange')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend(loc='upper left')
    plt.show(block=False)

    plt.pause(0.1) # Pause to update the plot