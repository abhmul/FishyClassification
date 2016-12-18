import matplotlib.pyplot as plt


def plot_history(history):

    plot_metric_dict(history.history)

def plot_metric_dict(metrics):

    plt.plot(metrics['val_loss'], 'o-')
    plt.plot(metrics['loss'], 'o-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Categorical Crossentropy')
    plt.title('Train Error vs Number of Iterations')

    plt.show()