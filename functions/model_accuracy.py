from matplotlib import pyplot as plt


def modelAccuracy(history):
    ax = plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    ax.legend(loc="upper left")
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    ax = plt.subplot(1,2,2)
    plt.plot(history.history['val_accuracy'], label='Vallidation accuracy')
    ax.legend(loc="upper left")
    fig.suptitle('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('No. epoch')
    plt.show()