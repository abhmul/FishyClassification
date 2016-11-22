import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history['val_loss'],'o-')
    plt.plot(history.history['loss'],'o-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Categorical Crossentropy')
    plt.title('Train Error vs Number of Iterations')

    plt.show()