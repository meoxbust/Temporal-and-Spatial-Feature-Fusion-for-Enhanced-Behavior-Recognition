from matplotlib import pyplot as plt


def modelLoss(history):
    fig = plt.figure(figsize=(14,6))
    ax = plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training loss')
    ax.legend(loc="upper left")
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    ax = plt.subplot(1,2,2)
    plt.plot(history.history['val_loss'], label='Vallidation loss')
    ax.legend(loc="upper left")
    fig.suptitle('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    plt.show()