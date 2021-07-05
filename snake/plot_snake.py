import numpy as np
import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot():
    data = np.load("snake_scores.npz")
    scores = data['scores']
    mean_scores = data['mean_scores']
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training data')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=True)


if __name__ == "__main__":
    plot()
